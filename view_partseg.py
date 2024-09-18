"""
Author: Benny
Date: Nov 2019
Modified by Eric Sep 2024
"""

import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

seg_classes = {"K": [0, 1, 2], "KT": [3, 4, 5, 6]}

seg_label_to_cat = {}  # {0:K, 1:K, ...6:KT}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def visualize_point_cloud_with_labels(points, labels):
    """使用 Open3D 可视化带有标签的点云"""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])  # 只显示 XYZ 坐标

    # 根据标签生成颜色
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(
        labels / (max_label if max_label > 0 else 1)
    )  # 使用颜色映射
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])  # 将颜色赋值给点云

    o3d.visualization.draw_geometries([point_cloud])


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("PointNet")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size in viewing"
    )
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument("--num_point", type=int, default=2048, help="point Number")
    parser.add_argument("--log_dir", type=str, default='pointnet2_part_seg_msg', help="experiment root")
    parser.add_argument(
        "--normal", action="store_true", default=False, help="use normals"
    )
    parser.add_argument(
        "--num_votes",
        type=int,
        default=3,
        help="aggregate segmentation scores with voting",
    )
    return parser.parse_args()


def main(args):
    """HYPER PARAMETER"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = "log/part_seg/" + args.log_dir

    root = "data/SteelShapeNet"

    TEST_DATASET = PartNormalDataset(
        root=root, npoints=args.num_point, split="view", normal_channel=args.normal
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    num_classes = 2
    num_part = 7

    """MODEL LOADING"""
    model_name = os.listdir(experiment_dir + "/logs")[0].split(".")[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + "/checkpoints/best_model.pth")
    classifier.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        seg_label_to_cat = {}  # {0:K, 1:K, ...6:KT}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()
        for batch_id, (points, label, target) in tqdm(
            enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9
        ):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = (
                points.float().cuda(),
                label.long().cuda(),
                target.long().cuda(),
            )
            points = points.transpose(2, 1)

            # 转换 PyTorch tensor 为 numpy 数组
            points_np = points[0, :3, :].cpu().numpy().T  # 获取第一批次并将其转换为 numpy

            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = (
                    np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                )

            # 渲染点云
            for i in range(cur_batch_size):
                points_np = points[i, :3, :].cpu().numpy().T  # 获取当前批次的点云
                pred_labels = cur_pred_val[i, :]  # 获取预测的分割标签
                visualize_point_cloud_with_labels(
                    points_np, pred_labels
                )  # 显示带有标签的点云


if __name__ == "__main__":
    args = parse_args()
    main(args)
