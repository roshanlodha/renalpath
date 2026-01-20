import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TumorDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train', model_name='resnet'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train' or 'val'/'test'.
            model_name (string): 'resnet', 'vit', or 'gsvit'.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.model_name = model_name

        # Prefer pre-computed label encoding from preprocessing for consistency across splits.
        if 'label_encoded' in self.data.columns:
            self.data['label_encoded'] = self.data['label_encoded'].astype(int)
            self.classes = None
            self.class_to_idx = None
        else:
            # Fallback: derive label indices from classes present in this CSV.
            self.classes = sorted(self.data['Class'].unique())
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def remove_borders(self, img):
        """
        Removes black borders using Otsu's thresholding and finding the largest contour.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            # Ensure ROI is valid
            if w > 0 and h > 0:
                return img[y:y+h, x:x+w]
        
        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path_rel = self.data.iloc[idx]['img_path']
        img_full_path = os.path.join(self.root_dir, img_path_rel)
        
        # 1. Load Image using OpenCV
        image = cv2.imread(img_full_path)
        if image is None:
             raise FileNotFoundError(f"Image not found at {img_full_path}")
        
        # 2. Border Removal
        image = self.remove_borders(image)
        
        # Convert BGR (OpenCV) to RGB (PIL) for Torchvision transforms compatibility
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)

        # 3. Padding & Center Crop logic
        w, h = image_pil.size
        target_size = 224*4
        
        pad_w = max(0, target_size - w)
        pad_h = max(0, target_size - h)
        
        if pad_w > 0 or pad_h > 0:
            padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
            image_pil = transforms.Pad(padding, fill=0, padding_mode='constant')(image_pil)
            
        image_pil = transforms.CenterCrop(target_size)(image_pil)

        # 4. Apply Transforms (Resize, Augmentations, etc.)
        if self.transform:
            image_pil = self.transform(image_pil)
            
        # image_pil is now a Tensor due to ToTensor() in transform
        
        # 5. Conditional Channel Flip for GSViT
        if self.model_name == 'gsvit':
            # Tensor is C, H, W (RGB). Convert to BGR by flipping channels.
            # RGB indices: 0, 1, 2 -> BGR indices: 2, 1, 0
            image_pil = image_pil[[2, 1, 0], :, :]

        # Label encoding
        if 'label_encoded' in self.data.columns:
            label = int(self.data.iloc[idx]['label_encoded'])
        else:
            label_name = self.data.iloc[idx]['Class']
            label = self.class_to_idx[label_name]

        return image_pil, label

def get_transforms(mode='train', model_name='resnet'):
    # Normalization stats
    if model_name == 'gsvit':
       mean = [0.5, 0.5, 0.5]
       std = [0.5, 0.5, 0.5]
    else:
       # ImageNet stats
       mean = [0.485, 0.456, 0.406]
       std = [0.229, 0.224, 0.225]

    resize_size = (224, 224) 

    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    return transform
