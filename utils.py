import os
import random
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplitlib.patches as patches

from config import config


def iou_width_height(bboxes1, bboxes2):
    """
    Intersections over union
    Args:
        bboxes1 (torch.tensor): width and height of the bbox
        bboxes2 (torch.tensor): width and height of the bbox
    Returns:
        iou (tensor): intersection over union of the corresponding bboxes
    """
    intersection = torch.min(bboxes1[..., 0], bboxes2[..., 0]) * torch.min(
        bboxes1[..., 1], bboxes2[..., 1])
    union = (
        bboxes1[..., 0] * bboxes1[..., 1] + bboxes2[..., 0] * bboxes2[..., 1]
        - intersection
    )
    iou = intersection / union
    return iou


def xyxy2cxcy(bbox):
    w = bbox[..., 2:3] - bbox[..., 0:1] + 1
    h = bbox[..., 3:4] - bbox[..., 1:2] + 1
    cx = (bbox[..., 2:3] + bbox[..., 0:1]) / 2
    cy = (bbox[..., 3:4] + bbox[..., 1:2]) / 2
    bbox[..., 0:1] = cx
    bbox[..., 1:2] = cy
    bbox[..., 2:3] = w
    bbox[..., 3:4] = h
    return bbox


def cxcy2xyxy(bbox):
    x0 = bbox[..., 0:1] - (bbox[..., 2:3] / 2 - 0.5)
    y0 = bbox[..., 1:2] - (bbox[..., 3:4] / 2 - 0.5)
    x1 = bbox[..., 0:1] + (bbox[..., 2:3] / 2 - 0.5)
    y1 = bbox[..., 1:2] + (bbox[..., 3:4] / 2 - 0.5)
    bbox[..., 0:1] = x0
    bbox[..., 1:2] = y0
    bbox[..., 2:3] = x1
    bbox[..., 3:4] = y1
    return bbox


def intersection_over_union(bbox_preds, bbox_labels, bbox_format="cxcy"):
    """
    Calculate intersection over union of predicted bboxes and target bboxes
    Args:
        bbox_preds (torch.tensor): predicted bboxes [batch_size, 4]
        bbox_labels (torch.tensor): ground truth bboxes [batch_size, 4]
        bbox_format (str): format of bboxes (center bbox and height with width: 'cxcy' or
            top left and bottom right coordinates: 'xyxy'
        )
    Returns:
        iou (torch.tensor): intersection over union between predicted bboxes
            and ground truth bboxes
    """
    box1 = bbox_preds.copy()
    box2 = bbox_labels.copy()
    if bbox_format == "cxcy":
        box1_x1 = bbox_preds[..., 0:1] - bbox_preds[..., 2:3] / 2
        box1_y1 = bbox_preds[..., 1:2] - bbox_preds[..., 3:4] / 2
        box1_x2 = bbox_preds[..., 2:3] + bbox_preds[..., 2:3] / 2
        box1_y2 = bbox_preds[..., 3:4] + bbox_preds[..., 3:4] / 2

        box2_x1 = bbox_labels[..., 0:1] - bbox_labels[..., 2:3] / 2
        box2_y1 = bbox_labels[..., 1:2] - bbox_labels[..., 3:4] / 2
        box2_x2 = bbox_labels[..., 2:3] + bbox_labels[..., 2:3] / 2
        box2_y2 = bbox_labels[..., 3:4] + bbox_labels[..., 3:4] / 2

    elif bbox_format == "xyxy":
        box1_x1 = box1[..., 0:1]
        box1_y1 = box1[..., 1:2]
        box1_x2 = box1[..., 2:3]
        box1_y2 = box1[..., 3:4]

        box2_x1 = box2[..., 0:1]
        box2_y1 = box2[..., 1:2]
        box2_x2 = box2[..., 2:3]
        box2_y2 = box2[..., 3:4]

    else:
        raise ValueError(
            f"Wrong 'bbox_format' value, available formats: 'cxcy', 'xyxy', but"
            f"got '{bbox_format}'"
        )

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    iou = intersection / (box1_area + box2_area - intersection)
    return iou



def non_max_suppression(bboxes, iou_threshold, threshold, bbox_format="xyxy"):
    """
    Non-maximum suppression (NMS) is a post-processing to eliminate
    duplicate detections and select bounding boxes

    Args:
        bboxes (list): predicted bboxes [class_pred, conf_score, x1, y1, x2, y2]
        iou_threshold (float): threshold to keep bboxes (not overlapping predictions)
        threshold (float): threshold for object confidence score
        bbox_format (str): bounding box format: cxcy, xyxy

    Returns:

    """
    # removing bboxes with score lower than the threshold
    bboxes = [box for box in bboxes if box[1] > threshold]
    # sort bboxes with descending confidence score
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []

    while bboxes:
        # selecting a bbox with a high confidence
        # and iteratively remove bboxes with lower confidence score
        # if they have the same class and high iou
        selected_bbox = bboxes.pop(0)

        bboxes = [
            box for box in bboxes
            if box[0] != selected_bbox[0] or intersection_over_union(
                torch.tensor(selected_bbox[2:]),
                torch.tensor(box[2:]),
                bbox_format=bbox_format
            ) < iou_threshold
        ] # removing bboxes with same class which has iou value higher than iou_threshold

        bboxes_after_nms.append(selected_bbox)

    return bboxes_after_nms


def mean_avearage_precision(
    pred_bboxes, true_bboxes, iou_threshold=0.5, bbox_format="cxcy", num_classes=20
):
    """
    Calculate mean Average Precision (mAP)
    Args:
        pred_bboxes (list): predicted bboxes [iamge_idx, cls, conf_score, x1, y1, x2, y2]
        true_bboxes (list): gt bboxes [iamge_idx, cls, conf_score, x1, y1, x2, y2]
        iou_threshold (float): threshold for correct predicted bbox
        bbox_format (str): bounding box type ['cxcy', 'xyxy']
        num_classes (int): number of classes

    Returns:
        mAP_score (float): mean Average Precision score (mAP)
    """
    average_precisions = []

    eps = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Iterate predictions and add only the
        # same target class
        for detection in pred_bboxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_bboxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # count bbox for each training samples
        count_bboxes = Counter([gt[0] for gt in ground_truths])


        for image_idx, num_bboxes in count_bboxes.items():
            count_bboxes[image_idx] = torch.zeros(num_bboxes)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # retrieve ground truth for the same image idx
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    bbox_format=bbox_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detection ground truth once
                if count_bboxes[detection[0]][best_gt_idx] == 0:
                    # add to the true positive and seen object
                    TP[detection_idx] = 1
                    count_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            # if IoU for detected object is lower than the threshold
            # the detection is false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP)
        FP_cumsum = torch.cumsum(FP)

        recalls = TP_cumsum / (total_true_bboxes + eps)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + eps)
        precisions = torch.cat([torch.tensor([1]), precisions])
        recalls = torch.cat([torch.tensor[0], recalls])
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    anchors,
    bbox_format="cxcy",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
):
    model.eval()
    train_idx = 0
    all_preds_bboxes = []
    all_true_bboxes = []
    for batch_idx, (images, labels) in enumerate(tqdm(loader)):
        images = images.to(device)

        with torch.no_grad():
            predictions = model(images)

        batch_size = images.size(0)
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            # get the feature size of predictions
            feat_size = predictions[i].size(2)
            anchor = torch.tensor([*anchors]).to(device) * feat_size
            scaled_bboxes = features2bboxes(
                predictions[i], anchor, feat_size=feat_size, is_preds=True
            )
            for idx, box in enumerate(scaled_bboxes):
                bboxes[idx] += box

        # one bbox for each label, not one bbox for each scale
        true_bboxes = features2bboxes(
            labels[2], anchor, feat_size=feat_size, is_preds=False
        )

        for idx in range(batch_size):
            nms_bboxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                bbox_format=bbox_format,
            )

            for nms_bbox in nms_bboxes:
                all_preds_bboxes.append([train_idx] + nms_bbox)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_bboxes.append([train_idx] + box)

            train_idx += 1
    model.train()
    return all_preds_bboxes, all_true_bboxes


def features2bboxes(predictions, anchors, feat_size, is_preds=True):
    """
    Scales model's predictions to input image
    Args:
        predictions (torch.tensor): predictions [batch_size, 3, feat_size, feat_size, 5 + num_classes]
            5: [conf_score, cx, cy, w, h]
        anchors (torch.tensor): anchors for predictions
        feat_size (int): feature size with corresponding height and width
        is_preds (bool): whether it is predicitions or ground truth bboxes

    Returns:
        converted_bboxes (torch.tensor): converted bounding boxes
            [batch_size, num_anchors, feat_size, feat_size, 5 + 1],
            5 + 1: [conf_score, cx, cy, h, w, cls]
    """
    batch_size = predictions.size(0)
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5] # removing the train_idx
    if is_preds:
        anchors = anchors.reshape(1, num_anchors, 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        conf_scores = torch.softmax(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        conf_scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(feat_size)
        .repeat(batch_size, num_anchors, feat_size, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )

    x = (box_predictions[..., 0:1] + cell_indices) / feat_size
    y = (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4)) / feat_size
    wh = box_predictions[2:4] / feat_size # [width, height]
    converted_bboxes = torch.cat([best_class, conf_scores, x, y, wh], dim=-1).reshape(
        batch_size, num_anchors * feat_size * feat_size, 6)
    return converted_bboxes.tolist()


def class_accuracy(model, loader, threshold, device):
    model.eval()
    total_cls_preds, correct_cls = 0, 0
    total_no_obj, correct_no_obj = 0, 0
    total_obj, correct_obj = 0, 0
    eps = 1e-16
    for idx, (images, labels) in enumerate(tqdm(loader)):
        images = images.to(device)
        with torch.no_grad():
            predictions = model(images)

        for i in range(3):
            labels[i] = labels[i].to(device)
            obj = labels[i][..., 0] == 1  # Iobj_i
            no_obj = labels[i][..., 0] == 0  # Iobj_i

            correct_cls += torch.sum(
                torch.argmax(predictions[i][..., 5:][obj], dim=-1) == labels[i][..., 5][obj]
            )
            total_cls_preds += torch.sum(obj)

            # object
            obj_preds = torch.sigmoid(predictions[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == labels[i][..., 0][obj])
            total_obj += torch.sum(obj)

            # no object
            correct_no_obj += torch.sum(obj_preds[no_obj] == labels[i][..., 0][no_obj])
            total_no_obj += torch.sum(no_obj)

    print(f"Class accuracy: {(correct_cls / (total_cls_preds + eps)) * 100:2f}%")
    print(f"No object accuracy: {(correct_no_obj / (total_no_obj + eps)) * 100:2f}%")
    print(f"Object accuracy: {(correct_obj / (total_obj + eps)) * 100:2f}%")
    model.train()



def save_checkpoint(model, optimizer, filename="ckpt.pth"):
    print("Saving checkpoint ...")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer, lr, device):
    print("Load checkpoint ...")
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Update old checkpoint learning rate with a new learning rate
    # in case of fine-tunning learning rate of checkpoint might be
    # too low to train
    for param_group in optimizer.param_groups():
        param_group["lr"] = lr


def get_loader(config):
    image_size = config.image_size
    train_csv_path = config.train_csv_path
    test_csv_path = config.test_csv_path

    train_dataset = YOLODataset(
        train_csv_path,
        transform=config.train_transform,
        feat_size=[
            image_size // 32, image_size // 16, image_size // 8
        ],
        image_dir=config.image_dir,
        label_dir=config.label_dir,
        anchors=config.anchors
    )
    eval_dataset = YOLODataset(
        train_csv_path,
        transform=config.test_transform,
        feat_size=[
            image_size // 32, image_size // 16, image_size // 8
        ],
        image_dir=config.image_dir,
        label_dir=config.label_dir,
        anchors=config.anchors
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transform,
        feat_size=[
            image_size // 32, image_size // 16, image_size // 8
        ],
        image_dir=config.image_dir,
        label_dir=config.label_dir,
        anchors=config.anchors
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False
    )
    return train_dataloader, eval_dataloader, test_dataloader


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False






















































