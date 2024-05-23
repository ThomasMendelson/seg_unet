import os
import cv2
import random
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Fluo_N2DH_SIM_PLUS(Dataset):
    def __init__(self, image_dir, resize, mask_dir=None, train_aug=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train_aug = train_aug
        self.images = os.listdir(image_dir)
        self.resize = resize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.mask_dir is not None:
            img_path = os.path.join(self.image_dir, self.images[index])
            mask_path = os.path.join(self.mask_dir, self.images[index].replace("t", "man_seg", 1))
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if len(image.shape) == 2:  # (height, width)
                image = np.expand_dims(image, axis=-1)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask = mask.astype(np.float32)
            # print("\nin dataset before augmentations = self.transform(image=image, mask=mask)")
            # print(f"image shape: {image.shape}, type: {image.dtype}, min value: {np.min(image)}, max value: {np.max(image)}")
            # print(f"mask shape: {mask.shape}, type: {mask.dtype}, min value: {np.min(mask)}, max value: {np.max(mask)}")

            if self.train_aug:
                crop_size  = int(min(image.shape[0],image.shape[1]) * random.uniform(0.8, 1.0))
                transform = get_train_transform(crop_size, self.resize)
            else:
                transform = get_val_transform(self.resize)
            augmentations = transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            return image, mask
        # else:
        #     img_path = os.path.join(self.image_dir, self.images[index])
        #     image = np.array(Image.open(img_path).convert("RGB"))
        #
        #     if self.transform is not None:
        #         augmentations = self.transform(image=image)
        #         image = augmentations["image"]
        #
        #     return image
    @staticmethod
    def detect_edges(mask, threshold=0.25):
        # Compute the gradients along rows and columns
        gradient_x = torch.gradient(mask, dim=1)
        gradient_y = torch.gradient(mask, dim=0)

        # Extract gradient components from the tuple
        gradient_x = gradient_x[0]
        gradient_y = gradient_y[0]

        gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
        masked_gradient_magnitude = gradient_magnitude * mask
        edge_mask = (masked_gradient_magnitude > threshold).to(torch.int)

        return edge_mask

    @staticmethod
    def split_mask(mask):
        three_classes_mask = torch.zeros_like(mask, dtype=torch.int32)
        for batch_idx in range(mask.size()[0]):
            unique_elements = torch.unique(mask[batch_idx].flatten())
            for element in unique_elements:
                if element != 0:
                    element_mask = (mask[batch_idx] == element).to(torch.int)
                    edges = Fluo_N2DH_SIM_PLUS.detect_edges(element_mask)
                    element_mask -= edges
                    three_classes_mask[batch_idx][edges == 1] = 1
                    three_classes_mask[batch_idx][element_mask == 1] = 2

        return three_classes_mask



def get_train_transform(crop_size, resize):
    train_transform = A.Compose(
        [
            A.ToFloat(max_value=65535.0),
            # A.Rotate(limit=35, p=1.0),
            A.RandomCrop(height=crop_size, width=crop_size),
            A.Resize(height=resize, width=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.0], std=[1.0], ),
            # A.FromFloat(max_value=65535.0),
            # A.Lambda(image=lambda x, **kwargs: x.astype(np.float32)),
            ToTensorV2(),
        ],
    )

    return train_transform

def get_val_transform(resize):
    val_transform = A.Compose(
        [
            A.ToFloat(max_value=65535.0),
            A.Resize(height=resize, width=resize),
            A.Normalize(mean=[0.0], std=[1.0], ),
            # A.FromFloat(max_value=65535.0),
            # A.Lambda(image=lambda x, **kwargs: x.astype(np.float32)),
            ToTensorV2(),
        ],
    )
    return val_transform


# import matplotlib.pyplot as plt
# def plot_img_befor_transform(image):
#     image = image.astype(np.float32) / np.iinfo(np.uint16).max
#     plt.imshow(image, cmap='gray')
#     plt.show()
#
# def plot_img_after_transform(image, mask):
#     fig, axs = plt.subplots(1, len(image), figsize=(15, 5))
#     fig2, axs2 = plt.subplots(1, len(mask), figsize=(15, 5))
#     for i, image in enumerate(image):
#         axs[i].imshow(image.squeeze()/torch.max(image), cmap='gray')
#         axs[i].axis("off")
#         print(f"min{i}: {torch.min(image)}, max{i}: {torch.max(image)}")
#     for i, mask in enumerate(mask):
#         axs2[i].imshow(mask.squeeze(), cmap='gray')
#         axs2[i].axis("off")
#         print(f"mask-min{i}: {torch.min(image)}, max{i}: {torch.max(image)}")
#
#     plt.show()



