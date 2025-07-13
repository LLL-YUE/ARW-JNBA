import numpy as np
import torch
import progressbar
import time
import os
from scipy import ndimage
import GeodisTK
import surface_distance as surfdist
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as ssim


def IoU(mask1, mask2):
    mask1, mask2 = (mask1>0.5).to(torch.bool), (mask2>0.5).to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()


def accuracy(mask1, mask2):  # mask1是gt
    mask1, mask2 = (mask1>0.5).to(torch.bool), (mask2>0.5).to(torch.bool)
    return torch.mean((mask1 == mask2).to(torch.float)).item()


def get_edge_points(img):
    dim = len(img.shape)
    if (dim==2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def surface_distance(point1, point2):
    """
    计算两个点之间的表面距离
    参数：
    - point1: 第一个点的坐标，格式为 (x, y)
    - point2: 第二个点的坐标，格式为 (x, y)
    返回值：
    两点之间的欧几里得距离
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def average_surface_distance_between_arrays(array1, array2):
    """
    计算两个二维数组之间的平均表面距离
    参数：
    - array1: 第一个二维数组，包含点坐标，每个点的格式为 (x, y)
    - array2: 第二个二维数组，包含点坐标，每个点的格式为 (x, y)
    返回值：
    平均表面距离
    """
    total_distance = 0
    num_pairs = 0

    # 遍历所有点对
    for point1 in array1:
        for point2 in array2:
            total_distance += surface_distance(point1, point2)
            num_pairs += 1

    # 计算平均值
    if num_pairs > 0:
        average_distance = total_distance / num_pairs
        return average_distance
    else:
        return 0


def HD(g, s):  # Hausdorff
    dist1 = directed_hausdorff(g, s)[0]
    dist2 = directed_hausdorff(s, g)[0]

    return max(dist1, dist2)


def precision_recall(mask_gt, mask):
    mask_gt, mask = mask_gt.to(torch.bool), mask.to(torch.bool)
    true_positive = torch.sum(mask_gt * (mask_gt == mask), dim=[-1, -2]).squeeze()
    mask_area = torch.sum(mask, dim=[-1, -2]).to(torch.float)
    mask_gt_area = torch.sum(mask_gt, dim=[-1, -2]).to(torch.float)

    precision = true_positive / mask_area
    precision[mask_area == 0.0] = 1.0

    recall = true_positive / mask_gt_area
    recall[mask_gt_area == 0.0] = 1.0

    return precision.item(), recall.item()


def F_score(p, r, betta_sq=1):
    f_scores = ((1 + betta_sq) * p * r) / (betta_sq * p + r)
    f_scores[f_scores != f_scores] = 0.0  # handle nans
    return f_scores


def F_max(precisions, recalls, betta_sq=1):
    F = F_score(precisions, recalls, betta_sq)  # 可以得到255个（在不同阈值下进行计算）F_score
    return F.mean(dim=0).max().item()


def MAE(mask1, mask2):
    # 将掩膜标准化到 [0, 1]
    mask1 = torch.clamp(mask1 / 255.0, 0, 1)
    mask2 = torch.clamp(mask2 / 255.0, 0, 1)
    mae = torch.abs(mask1 - mask2).mean().item()
    return mae


def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

    return score


def S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg
    return Q


def _centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        X = torch.eye(1) * round(cols / 2)
        Y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        i = torch.from_numpy(np.arange(0, cols)).float()
        j = torch.from_numpy(np.arange(0, rows)).float()
        X = torch.round((gt.sum(dim=0) * i).sum() / total)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total)
    return X.long(), Y.long()

def _divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4

def _dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB

def _ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q

def _S_region(pred, gt):
    X, Y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q


def S_measure(mask1, mask2, alpha=0.5):
    gt = torch.clamp(mask1 / 255.0, 0, 1)
    pred = torch.clamp(mask2 / 255.0, 0, 1)
    y = gt.mean()
    if y == 0:
        x = pred.mean()
        Q = 1.0 - x
    elif y == 1:
        x = pred.mean()
        Q = x
    else:
        gt[gt>=0.5] = 1
        gt[gt<0.5] = 0
        Q = alpha * S_object(pred, gt) + (1-alpha) * _S_region(pred, gt)
        if Q.item() < 0:
            Q = torch.FloatTensor([0.0])
    return Q.item()


def E_measure(mask1, mask2):
    gt = torch.clamp(mask1 / 255.0, 0, 1)
    pred = torch.clamp(mask2 / 255.0, 0, 1)
    scores = torch.zeros(255)
    scores += eval_e(pred, gt, 255)
    return scores.max().item()

def eval_e(y_pred, y, num):
    score = torch.zeros(num)
    thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_pred_th = (y_pred >= thlist[i]).float()
        fm = y_pred_th - y_pred_th.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
    return score

@torch.no_grad()
def metrics(pred, gt, stats=(IoU, accuracy, F_max, HD, MAE, S_measure, E_measure), prob_bins=255):
    avg_values = {}
    precisions = []
    recalls = []
    out_dict = {}

    nb_sample = len(gt)
    p = progressbar.ProgressBar()
    for step in p(range(nb_sample)):
        prediction, mask = torch.from_numpy(pred[step]), torch.from_numpy(gt[step])
        for metric in stats:
            method = metric.__name__
            if method not in avg_values and metric != F_max:
                avg_values[method] = 0.0

            if metric in (IoU, accuracy, HD, MAE, S_measure, E_measure):
                value = metric(mask, prediction)
                avg_values[method] += value
            else:
                p, r = [], []
                splits = 2.0 * prediction.mean(dim=0) if prob_bins is None else \
                    np.arange(0.0, 1.0, 1.0 / prob_bins)

                for split in splits:
                    pr = precision_recall(mask, prediction > split)
                    p.append(pr[0])
                    r.append(pr[1])
                precisions.append(p)
                recalls.append(r)
        time.sleep(0.01)

    for metric in stats:
        method = metric.__name__
        if metric == F_max:
            out_dict[method] = F_max(torch.tensor(precisions), torch.tensor(recalls))
        else:
            out_dict[method] = avg_values[method] / nb_sample

    return out_dict