import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from tqdm import tqdm
from PIL import Image

import albumentations as A
from model import UNET
from albumentations.pytorch import ToTensorV2
from dataset import Fluo_N2DH_SIM_PLUS

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_test_predictions_as_imgs,
    apply_color_map,
)

# Hyperparameters
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 1
NUM_WORKERS = 0
# IMAGE_HEIGHT = 512  # 690 originally
# IMAGE_WIDTH = IMAGE_HEIGHT  # 628 originally
INPUT_SIZE = 512
CLASS_WEIGHTS = [0.15, 0.6, 0.25]
PIN_MEMORY = False
LOAD_MODEL = False
WANDB_TRACKING = False
TRAIN_IMG_DIR = "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/02"
TRAIN_MASK_DIR = "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/02_ERR_SEG"
VAL_IMG_DIR = "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/01"
VAL_MASK_DIR = "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/01_ERR_SEG"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0.0
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = Fluo_N2DH_SIM_PLUS.split_mask(targets).long().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def evaluate_fn(loader, model, loss_fn):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device=DEVICE)

            targets = Fluo_N2DH_SIM_PLUS.split_mask(targets).long().to(device=DEVICE)
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    if WANDB_TRACKING:
        wandb.login(key="12b9b358323faf2af56dc288334e6247c1e8bc63")
        wandb.init(project="seg_unet")

    model = UNET(in_channels=1, out_channels=3).to(DEVICE)
    class_weights = torch.FloatTensor(CLASS_WEIGHTS).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        INPUT_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("../my_checkpoint.pth.tar", map_location=torch.device(DEVICE)), model)

    # check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"epoch: [{epoch + 1}/{NUM_EPOCHS}]")

        train_loss = train_fn(train_loader, model, optimizer, criterion, scaler)
        val_loss = evaluate_fn(val_loader, model, criterion)

        if WANDB_TRACKING:
            wandb.log({"Train Loss": train_loss, "Val Loss": val_loss})

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename="../my_checkpoint.pth.tar")

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, device=DEVICE)
    if WANDB_TRACKING:
        wandb.finish()




# def t_images():
#     test_transform = A.Compose(
#         [
#             A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
#             A.Normalize(mean=[0.0], std=[1.0], ),
#             ToTensorV2(),
#         ],
#     )
#     # test_transform = transforms.Compose([
#     #     transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
#     #     transforms.ToTensor(),
#     # ])
#     model = UNET(in_channels=1, out_channels=1).to(DEVICE)
#
#     test_ds = Fluo_N2DH_SIM_PLUS(
#         image_dir=r"C:\BGU\u-net_seg\seg_unet\Fluo-N2DH-SIM+_tset-datasets\Fluo-N2DH-SIM+\02",
#         transform=test_transform,
#     )
#
#     test_loader = DataLoader(
#         test_ds,
#         batch_size=1,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY,
#         shuffle=False,
#     )
#
#     load_checkpoint(torch.load("../my_checkpoint_last_one.pth.tar", map_location=torch.device(DEVICE)), model)
#
#     save_test_predictions_as_imgs(
#         test_loader, model, folder=r"C:\BGU\u-net_seg\seg_unet\Fluo-N2DH-SIM+_tset-datasets\Fluo-N2DH-SIM+\saved\02",
#         device=DEVICE, type="test"
#     )


if __name__ == "__main__":
    main()
    # t_acc()
