import random
import torch
import torch.nn as nn

from utils import intersection_over_union


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 1
        self.lambda_no_obj = 10
        self.lambda_obj = 1
        self.lambda_bbox = 10

        self.eps = 1e-16

    def forward(self, predictions, targets, anchors):
        obj = targets[..., 0] == 1
        no_obj = targets[..., 0] == 0

        # No object loss
        no_object_loss = self.bce(
            predictions[..., 0:1][no_obj], targets[..., 0:1][no_obj]
        )

        # object loss
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        bbox_preds = torch.cat(
            [
                self.sigmoid(predictions[..., 1:3]),
                torch.exp(predictions[..., 3:5] * anchors)
            ], dim=-1
        )
        iou = intersection_over_union(
            bbox_preds[obj], targets[..., 1:5][obj]
        ).detach()
        object_loss = self.mse(
            self.sigmoid(predictions[..., 0:1][obj]),
            iou * targets[..., 0:1][obj]
        )

        # bbox coordinate
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        targets[..., 3:5] = torch.log(
            targets[..., 3:5] / anchors + self.eps
        )
        bbox_loss = self.mse(predictions[..., 1:5][obj], targets[..., 1:5][obj])

        # class loss
        class_loss = self.entropy(
            predictions[..., 5:][obj], targets[..., 5:][obj].type(predictions.dtype)
        )

        return (
            self.lambda_bbox * bbox_loss
            + self.lambda_obj * object_loss
            + self.lambda_no_obj * no_object_loss
            + self.lambda_class * class_loss
        )