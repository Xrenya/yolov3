from typing import List
from dataclasses import dataclass
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import seed_everything


SEED: int = 42
IMAGE_SIZE: int = 224
BATCH_SIZE: int = 16
NUM_EPOCHS: int = 10
ROOT_PATH: str = os.path.join(os.environ.get("ROOT_PATH"))
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

# Train transformation
scale = 1.1
train_transformation = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT)
            ], p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ]
)

# Test/Eval transformation
test_transformation = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ]
)

# class for object detection
classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]


config = Config(
    dataset="PASCAL_VOC",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    num_workers=1,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    num_classes=20,
    learning_rate=1e-5,
    weight_decay=1e-4,
    num_epochs=NUM_EPOCHS,
    conf_threshold=0.05,
    map_iou_threshold=0.5,
    nms_iou_threshold=0.5,
    feat_size=[],
    pin_memory=True,
    load_model=True,
    save_model=False,
    checkpoint="checkpoint.pth",
    image_dir="/PASCAL_VOC/images/",
    label_dir="/PASCAL_VOC/labels/",
    anchors=ANCHORS,
    train_transformation=train_transformation,
    test_transformation=test_transformation,
    classes=classes,
)
