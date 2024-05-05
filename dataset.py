import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class Fluo_N2DH_SIM_PLUS(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.mask_dir is not None:
            img_path = os.path.join(self.image_dir, self.images[index])
            mask_path = os.path.join(self.mask_dir, self.images[index].replace("t", "mask", 1))
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask > 0] = 1.0
            # image = np.transpose(image, (2, 0, 1))
    
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
            print(f"in dataset, image: {image.size()} mask: {mask.size()}")
            return image, mask
        else:
            img_path = os.path.join(self.image_dir, self.images[index])
            image = np.array(Image.open(img_path).convert("RGB"))
            # image = np.transpose(image, (2, 0, 1))
            # image = np.transpose(image, (0, 3, 1, 2))
            if self.transform is not None:
                augmentations = self.transform(image=image)
                image = augmentations["image"]
            # print(f"in dataset, image: {image.size()} ")
            return image
    # def __getitem__(self, index):
    #     img_name = self.images[index]
    #     img_path = os.path.join(self.image_dir, img_name)
    #     image = Image.open(img_path).convert("RGB")
    #
    #     if self.mask_dir is not None:
    #         mask_name = img_name.replace("t", "mask", 1)
    #         mask_path = os.path.join(self.mask_dir, mask_name)
    #         mask = Image.open(mask_path).convert("L")
    #         mask = np.array(mask, dtype=np.float32)
    #         mask[mask > 0] = 1.0
    #
    #     if self.transform is not None:
    #         # Convert PIL Image to NumPy array for mask
    #         if self.mask_dir is not None:
    #             mask_pil = Image.fromarray(mask)
    #             image = self.transform(image)
    #             mask = self.transform(mask_pil)
    #
    #         else:
    #             image = self.transform(image)
    #
    #     if self.mask_dir is not None:
    #         return image, mask
    #     else:
    #         return image

