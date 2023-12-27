import torch
import torch.optim as optim
from tqdm import tqdm

from config import config
from model import YOLOv3
from utils import (
    mean_average_precision,
    seed_everything,
    load_checkpoint,
    save_checkpoint,
    class_accuracy,
    features2bboxes,
    get_evaluation_bboxes,
    mean_avearage_precision,
    non_max_suppression,
    intersection_over_union,
    iou_width_height
)
from loss import Loss


def train(config, train_dataloader, model, optimizer, loss_fn, scaler, scaled_anchors):
    losses = []
    iterator = tqdm(train_dataloader, leave=True)
    for idx, (images, target) in enumerate(iterator):

        optimizer.zero_grad()

        images = images.to(config.device)
        y0, y1, y2 = (
            target[0].to(config.device),
            target[1].to(config.device),
            target[2].to(config.device)
        )

        with torch.cuda.amp.autocast():
            predictions = model(images)
            loss = (
                loss_fn(predictions[0], y0, scaled_anchors[0])
                + loss_fn(predictions[1], y1, scaled_anchors[1])
                + loss_fn(predictions[2], y2, scaled_anchors[2])
            )

        losses.append(loss.detach().cpu().item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(losses) / len(losses)
        iterator.set_postfix(loss=mean_loss)


def main():
    model = YOLOv3(num_classes=config.num_classes).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.leraning_rate, weight_decay=config.weight_decay)
    loss_fn = Loss()
    scaler = torch.cuda.amp.GradScaler()

    train_dataloader, eval_dataloader, test_dataloader = get_loader(config)

    scaled_anchors = torch.tensor(config.anchors)