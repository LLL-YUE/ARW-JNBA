"""
Main experiment file. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import os
import argparse

import torch
import datetime
import numpy as np

from tqdm import tqdm
from PIL import Image

from networks import get_model
from object_discovery import ncut
from torchvision import transforms as T
import jenkspy
import metric
import torch.nn.functional as F
import cv2
from sceneRadianceCLAHE import RecoverCLAHE
from crf import densecrf


NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# Image transformation applied to all images
transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

def classify1(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1

def goodness_of_variance_fit(array, classes):
    # get the break points
    classes = jenkspy.jenks_breaks(array, classes)
    # do the actual classification
    classified = np.array([classify1(i, classes) for i in array])
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


def precision_recall_curve(pred, gt, thresholds=np.linspace(0, 1, 100)):
    precisions = []
    recalls = []

    for thresh in thresholds:
        TP, FP, FN = 0, 0, 0  # 初始化总的 TP, FP, FN

        for i in range(len(pred)):  # 遍历所有图像
            print("Prediction range: ", np.min(pred[i]), np.max(pred[i]))
            binary_pred = (pred[i] >= thresh).astype(int)  # 当前图像的二值化结果
            binary_gt = (gt[i] == 255).astype(int)  # 对应的真实标签
            # 逐元素计算 TP, FP, FN
            TP += np.sum((binary_pred == 1) & (binary_gt == 1))
            FP += np.sum((binary_pred == 1) & (binary_gt == 0))
            FN += np.sum((binary_pred == 0) & (binary_gt == 1))

        # 避免除零错误
        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--arch",
        default="vit_base",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "resnet50",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="SUIM",
        type=str,
        choices=[None, "SUIM", "UFO", "USOD"],
        help="Dataset name.",
    )
    
    parser.add_argument(
        "--save-feat-dir",
        type=str,
        default=None,
        help="if save-feat-dir is not None, only computing features and save it into save-feat-dir",
    )
    
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")     # 若触发参数no_hard则为true，否则为false

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["attn", "pred", "all", None],
        default=all,
        help="Select the different type of visualizations.",
    )

    # TokenCut parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v"],
        help="Which features to use",
    )
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )
    parser.add_argument("--resize", type=int, default=None, help="Resize input image to fix size")
    parser.add_argument("--tau", type=float, default=0.2, help="Tau for seperating the Graph.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Eps for defining the Graph.")
    parser.add_argument("--no-binary-graph", action="store_true", default=False, help="Generate a binary graph where edge of the Graph will binary. Or using similarity score as edge weight.")

    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)

    args = parser.parse_args()

    if args.image_path is not None:
        args.no_evaluation = True
        args.dataset = None

    # -------------------------------------------------------------------------------------------------------
    # Dataset
    if args.dataset == 'SUIM':
        args.img_dir = './datasets/SUIM/images'
        args.gt_dir = './datasets/SUIM/masks'
    elif args.dataset == 'UFO':
        args.img_dir = './datasets/UFO-120/images'
        args.gt_dir = './datasets/UFO-120/masks'
    elif args.dataset == 'USOD':
        args.img_dir = './datasets/USOD/images'
        args.gt_dir = './datasets/USOD/masks'
    elif args.dataset is None:
        args.gt_dir = None

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        img_list = [args.image_path]
    else:
        img_list = sorted(os.listdir(args.img_dir))

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args.arch, args.patch_size, device)

    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    if args.dinoseg:
        # Experiment with the baseline DINO-seg
        if "vit" not in args.arch:
            raise ValueError("DINO-seg can only be applied to tranformer networks.")
        exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
    else:
        # Experiment with TokenCut 
        exp_name = f"TokenCut-{args.arch}"
        if "vit" in args.arch:
            exp_name += f"{args.patch_size}_{args.which_features}"

    print(f"Running TokenCut on the dataset {args.dataset} (exp: {exp_name})")

    # Visualization 
    if args.visualize:
        vis_folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)
        
    if args.save_feat_dir is not None:
        os.mkdir(args.save_feat_dir)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    mask_lost = []
    gt = []
    for img_name in tqdm(img_list):  # 对每张图片进行处理
        # ------------ IMAGE PROCESSING -------------------------------------------
        if args.image_path is not None:
            img_pth = img_name
            img_name = img_name.split("/")[-1]  # 带后缀的图片名
        else:
            img_pth = os.path.join(args.img_dir, img_name)

        # Get the name of the image
        im_name = img_name.split(".")[0]   # 不带后缀的图片名
        img = Image.open(img_pth).convert("RGB")
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        sceneRadiance = RecoverCLAHE(img)
        img = Image.fromarray(cv2.cvtColor(sceneRadiance, cv2.COLOR_BGR2RGB))
        trans_img = img.copy()
        img = transform(img)

        init_image_size = img.shape

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        # Move to gpu
        if device == torch.device('cuda'):
            img = img.cuda(non_blocking=True)   # 把数据迁移到GPU
        # Size for transformers
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            if "vit" in args.arch:
                # Store the outputs of qkv layer from the last attention layer
                feat_out = {}

                def hook_fn_forward_qkv(module, input, output):  # 提取qkv模块作为输出
                    feat_out["qkv"] = output
                model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
                # ["block"][-1]表示只有最后一个block会调用hook_fn_forward_qkv，因为输出的就是最后一层注意力层
                # register_forward_hook钩子机制，导出需要的中间变量
                try:
                    # Forward pass in the model
                    attentions = model.get_last_selfattention(img[None, :, :, :])
                except:
                    continue

                # Scaling factor
                scales = [args.patch_size, args.patch_size]

                # Dimensions
                nb_im = attentions.shape[0]  # Batch size
                nh = 6  # Number of heads
                nb_tokens = attentions.shape[2]  # Number of tokens

                # Extract the qkv features of the last attention layer
                qkv = (feat_out["qkv"].reshape(nb_im, nb_tokens, 3, nh, -1 // nh).permute(2, 0, 3, 1, 4))
                q, k, v = qkv[0], qkv[1], qkv[2]
                k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                feat = k[:, 1:, :].reshape(k.shape[0], w_featmap, h_featmap, k.shape[2]).permute(0, 3, 1, 2)

                # Modality selection
                if args.which_features == "k":    # key表示待查询的特征
                    feats = k
                elif args.which_features == "q":  # q表示要查询的特征
                    feats = q
                elif args.which_features == "v":  # v表示待查询的特征对应的下标
                    feats = v
                if args.save_feat_dir is not None:
                    np.save(os.path.join(args.save_feat_dir, im_name.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy')), feats.cpu().numpy())
                    continue
            else:
                raise ValueError("Unknown model.")

        if "vit" in args.arch:
            foreground, seed, bins, eigenvector = ncut(feats, [w_featmap, h_featmap], scales, init_image_size, args.tau, args.eps, im_name=im_name, no_binary_graph=args.no_binary_graph)

            image = Image.open(img_pth)
            image = image.convert("RGB")
            mask = torch.from_numpy(foreground).to('cuda')
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=[init_image_size[-2], init_image_size[-1]], mode='nearest').squeeze().cpu().numpy()  # 恢复尺寸
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            sceneRadiance = RecoverCLAHE(image)
            image = Image.fromarray(cv2.cvtColor(sceneRadiance, cv2.COLOR_BGR2RGB))
            mask = densecrf(np.array(image), mask)
            mask_lost.append(mask)

        # ------------ GROUND-TRUTH -------------------------------------------
        if args.gt_dir is not None:
            if args.dataset == 'SUIM':
                mask_gt = np.array(
                    Image.open(os.path.join(args.gt_dir, img_name.replace('.jpg', '.bmp'))).convert('L'))
            elif args.dataset == 'UFO' or args.dataset == 'USOD':
                mask_gt = np.array(
                    Image.open(os.path.join(args.gt_dir, img_name)).convert('L'))
            gt.append(mask_gt)

    # Evaluate
    if args.gt_dir is not None and args.image_path is None:
        print('evaluation:')
        value = metric.metrics(mask_lost, gt)
        print(value)
        print('\n')