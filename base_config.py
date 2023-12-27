from typing import Optional, List
from dataclasses import dataclass
import cv2
import torch
import albumentations as A

from utils import seed_everything


@dataclass
class Config:
    dataset: str
    device: torch.device
    num_workers: int
    batch_size: int
    image_size: int
    num_classes: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    conf_threshold: float
    map_iou_threshold: float
    nms_iou_threshold: float
    feat_size: List[int]
    pin_memory: bool
    load_model: bool
    save_model: bool
    checkpoint: Optional[str]
    image_dir: str
    label_dir: str
    anchors: List[List[float]]
    train_transformation: A.Compose
    test_transformation: A.Compose
    classes: List[str]


