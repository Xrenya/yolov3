import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from config import config
from utils import (
    features2bboxes,
    iou_width_height,
    non_max_suppresion as nms
)


class YoloDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        label_dir,
        anchors,
        image_size=416,
        feat_size=[13, 26, 52],
        classes=20,
        transform=None
    ):
        self.anno = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.feat_sizes = feat_size
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = len(self.anchors)
        self.num_anchors_per_scale = self.num_anchors // len(self.feat_sizes)
        self.classes = classes
        self.ignore_iou_threshold = 0.5

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, item):
        label_path = os.path.join(self.label_dir, self.anno.iloc[index, 1])
        bboxes = np.roll(
            np.loadtxt(fname=label_path, delimiter=" ", ndmin=2),
            4,
            axis=1
        ).tolist()
        image_path = os.path.join(self.image_dir, self.anno.iloc[index, 0])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        if self.transform:
            aug = self.transform(image=image, bboxes=bboxes)
            image = aug["image"]
            bboxes = aug["bboxes"]

        targets = [
            torch.zeros(
                (self.num_anchors // len(self.feat_sizes), feat_size, feat_size)
            ) for feat_size in self.feat_sizes
        ]
        for bbox in bboxes:
            iou_anchors = iou_width_height(torch.tensor(bbox[2:4]), self.anchors)
            anchors_indices = iou_anchors.argsort(descending=True, dim=0)
            cx, cy, width, height, label = bbox
            has_anchor = [False] * len(self.feat_sizes)
            for anchor_idx in anchors_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                feat_size = self.feat_sizes[scale_idx]
                i, j = int(feat_size * y), int(feat_size * x)
                anchor_is_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_is_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = feat_size * x - j, feat_size * y - i
                    width_cell, height_cell = (
                        width * feat_size,
                        height_cell * feat_size
                    )
                    bbox_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = bbox_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(label)
                    has_anchor[scale_idx] = True

                elif not anchor_is_taken and iou_anchors[anchor_idx] > self.ignore_iou_threshold:
                    # ignore prediction
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)































