import os

import torch
import torchvision
from dataset import Fluo_N2DH_SIM_PLUS
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=0,
    pin_memory=True
):
    train_ds = Fluo_N2DH_SIM_PLUS(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = Fluo_N2DH_SIM_PLUS(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    print("=> checking accuracy")
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # y = y.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}%"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder=r"C:\BGU\seg_unet\Fluo-N2DH-SIM+_training-datasets\saved", device="cuda"
):
    print("=> saving images")
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        # os.makedirs("C:\\BGU\\u-net_seg\\Fluo-N2DH-SIM+_training-datasets\\saved\\", exist_ok=True)
        # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
        # torchvision.utils.save_image(y, f"{folder}/{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")

    model.train()


def save_test_predictions_as_imgs(
    loader, model, folder=r"C:\BGU\seg_unet\Fluo-N2DH-SIM+_training-datasets\saved", device="cuda", type="train"
):
    model.eval()
    if type == "train":

        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()

            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
            # torchvision.utils.save_image(y, f"{folder}/{idx}.png")

    else:
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        for idx, x in enumerate(loader):
            x = x.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()

            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            # x = torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0))
            # torchvision.utils.save_image(x, f"{folder}/x_{idx}.png")
            # num_correct += (preds == x).sum()
            # num_pixels += torch.numel(preds)
            # dice_score += (2 * (preds * x).sum()) / ((preds + x).sum() + 1e-8)

        # print(
        #     f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}%"
        # )
        # print(f"Dice score: {dice_score / len(loader)}")
    model.train()
