"""
Main functions for applying Normalized Cut.
Code adapted from LOST: https://github.com/valeoai/LOST
"""
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from scipy import ndimage
import jenkspy

def goodness_of_variance_fit(array, classes):
    # get the break points
    classes = jenkspy.jenks_breaks(array, classes)
    # do the actual classification
    classified = np.array([classify(i, classes) for i in array])
    # max value of zones
    maxz = max(classified)
    # nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]
    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)
    # sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]
    # sum of squared deviations of class means
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])
    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam
    return gvf


def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1


def ncut(feats, dims, scales, init_image_size, tau=0, eps=1e-5, im_name='', no_binary_graph=False):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """

    feats = feats[0, 1:, :]
    feats = F.normalize(feats, p=2)
    A = (feats @ feats.transpose(1, 0))
    A = A.cpu().numpy()
    if no_binary_graph:
        A[A<tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)

    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
  
    # Print second and third smallest eigenvector 
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1, 2])  # 求解广义特征值问题：subset_by_index表示仅返回第二小到第三小的特征值
    eigenvec = np.copy(eigenvectors[:, 0])

    # Using average point to compute bipartition（分割前景和背景）
    second_smallest_vec = eigenvectors[:, 0]  # 第二小特征向量

    gvf = 0.0
    classes = 2
    while gvf < 0.8 and classes < 20:  # 0.8
        gvf = goodness_of_variance_fit(second_smallest_vec, classes)
        classes += 1
    seed = np.argmax(np.abs(second_smallest_vec))  # 找到最大的特征向量绝对值
    if second_smallest_vec[seed] > 0:  # 说明特征向量越大越显著
        breaks = jenkspy.jenks_breaks(second_smallest_vec, n_classes=classes)  # 返回的是断点（包括最大值和最小值）
        bipartition = second_smallest_vec > breaks[1]  # 分类
    else:  # 说明特征向量越小越显著
        eigenvec = eigenvec * -1
        breaks = jenkspy.jenks_breaks(second_smallest_vec, n_classes=classes)  # 返回的是断点（包括最大值和最小值）
        bipartition = second_smallest_vec < breaks[classes-1]  # 分类

    bipartition = bipartition.reshape(dims).astype(float)
    return bipartition, seed, None, eigenvec.reshape(dims)  # , mask1, mask2, mask3



def detect_box(bipartition, initial_im_size=None, scales=None, principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    s = [[1, 1, 1],   # 八连接
         [1, 1, 1],
         [1, 1, 1]]
    objects, num_objects = ndimage.label(bipartition, structure=s)  # 标记连接成分：输入中的非零值被视为特征，零值被视为背景
    # 计算每个区域的尺寸
    max_area = 0
    for i in range(1, num_objects+1):
        area = ndimage.sum(bipartition, objects, index=i)
        if area > max_area:
            max_area = area
            cc = i  # 这里的cc表示面积最大的区域对应的标签

    # cc = objects[np.unravel_index(seed, dims)]  # 这里的cc表示包含最大特征向量的区域对应的标签

    if principle_object:
        mask = np.where(objects == cc)
       # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]
         
        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])
        
        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError