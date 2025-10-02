import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np

class SynSkinDataset(Dataset):
    def __init__(self, csv_path, annotations_path, dataset_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.annotations = pd.read_csv(annotations_path)
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.color_names = ["light_brown", "dark_brown", "black", "blue_gray", "red", "white"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row["ID"]

        img_path = os.path.join(self.dataset_dir, "SynSkin", "images", img_id, f"{img_id}_synthetic.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load all 6 masks (one binary mask for each color)
        masks = []
        for color in self.color_names:
            mask_path = os.path.join(self.dataset_dir, "SynSkin", "images", img_id, f"{img_id}_color_mask_{color}.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype('uint8')  # Binarize
            masks.append(mask)

        # Border mask
        border_mask_path = os.path.join(self.dataset_dir, "SynSkin", "images", img_id, f"{img_id}_border_mask.png")
        border_mask = cv2.imread(border_mask_path, cv2.IMREAD_GRAYSCALE)
        border_mask = (border_mask > 127).astype('uint8')  # Binarize

        # Lesion mask
        lesion_mask_path = os.path.join(self.dataset_dir, "SynSkin", "images", img_id, f"{img_id}_lesion_mask.png")
        lesion_mask = cv2.imread(lesion_mask_path, cv2.IMREAD_GRAYSCALE)
        lesion_mask = (lesion_mask > 127).astype('uint8')  # Binarize

        labels = torch.tensor(self.annotations.loc[self.annotations.ID == img_id].values[0][1:].astype(float))  # [0,1,0,1,...] (6 classes)

        # Prepare mask dict for Albumentations
        additional_masks = {f'mask{i}': masks[i] for i in range(6)}

        if self.transform:
            augmented = self.transform(
                image=image,
                **additional_masks,
                border_mask=border_mask,
                lesion_mask=lesion_mask
            )
            image = augmented['image']
            color_masks = torch.stack([augmented[f'mask{i}'] for i in range(6)], dim=0)
            border_mask = augmented['border_mask']
            lesion_mask = augmented['lesion_mask']
        else:
            image = ToTensorV2()(image=image)['image']
            color_masks = torch.stack([ToTensorV2()(image=mask)['image'][0] for mask in masks], dim=0)
            border_mask = ToTensorV2()(border_mask=border_mask)['border_mask']
            lesion_mask = ToTensorV2()(lesion_mask=lesion_mask)['lesion_mask']

        return {
            "image": image, 
            "color_masks": color_masks, 
            "border_mask": border_mask,
            "lesion_mask": lesion_mask,
            "labels": labels,
            "id": img_id
        }
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transforms import get_train_transform, get_val_transform
    from utils_dataset import plot_image_and_masks

    example_dataset = SynSkinDataset(
        csv_path="/Users/cristianopatricio/Documents/synskin-dataset/SynSkin/metadata/synskin_train_split.csv",
        annotations_path="/Users/cristianopatricio/Documents/synskin-dataset/SynSkin/metadata/annotations_synskin_colors.csv",
        dataset_dir="/Users/cristianopatricio/Documents/synskin-dataset",
        transform=get_val_transform()
    )

    example_dataloader = DataLoader(
        example_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )

    for batch in example_dataloader:
        print(">>> Batch summary <<<<")
        print(f"ID: {batch['id'][0]}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Color Masks shape: {batch['color_masks'].shape}")
        print(f"Border Mask shape: {batch['border_mask'].shape}")
        print(f"Lesion Mask shape: {batch['lesion_mask'].shape}")
        print(f"Labels: {batch['labels'].flatten().int().tolist()}")

        # Plot batch
        plot_image_and_masks(batch)
        break