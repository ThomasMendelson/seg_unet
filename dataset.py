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
    
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
                # mask = Fluo_N2DH_SIM_PLUS.split_mask(mask)

            return image, mask
        else:
            img_path = os.path.join(self.image_dir, self.images[index])
            image = np.array(Image.open(img_path).convert("RGB"))

            if self.transform is not None:
                augmentations = self.transform(image=image)
                image = augmentations["image"]

            return image

    @staticmethod
    def detect_edges(mask, threshold=0.25):
        # Compute the gradients along rows and columns
        gradient_x = np.gradient(mask.astype(float), axis=1)
        gradient_y = np.gradient(mask.astype(float), axis=0)

        # Extract gradient components from the tuple
        gradient_x = gradient_x[0]
        gradient_y = gradient_y[0]

        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        masked_gradient_magnitude = gradient_magnitude * mask
        edge_mask = (masked_gradient_magnitude > threshold).astype(int)

        return edge_mask

    @staticmethod
    def split_mask(mask):
        three_classes_mask = torch.zeros_like(mask, dtype=torch.int32)
        unique_elements = torch.unique(mask.flatten())
        for element in unique_elements:
            if element != 0:
                element_mask = (mask == element).to(torch.int)
                edges = Fluo_N2DH_SIM_PLUS.detect_edges(element_mask)
                element_mask -= edges
                three_classes_mask[edges == 1] = 1
                three_classes_mask[element_mask == 1] = 2

        return three_classes_mask



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

