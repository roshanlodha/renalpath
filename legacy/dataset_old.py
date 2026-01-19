import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2

class TumorDataset(Dataset):
    def __init__(self, csv_file, transform=None, is_gsvit=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_gsvit (bool): If True, converts images to BGR format.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.is_gsvit = is_gsvit

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['path']
        label = self.data.iloc[idx]['label_encoded']
        
        # Load image
        # Images are already saved as RGB/BGR in preprocessing but let's load safely
        # Using PIL for consistency with torchvision transforms
        image = Image.open(img_path).convert('RGB')
        
        if self.is_gsvit:
            # GSViT expects BGR. 
            # If we use standard torchvision transforms, they expect RGB PIL or Tensor.
            # We should apply standard transforms first, then convert to BGR tensor if needed.
            # OR convert to BGR numpy, then to Tensor.
            # The prompt says: "Input images were preprocessed to match GSViT’s expected BGR format by flipping the first and third channels."
            # It also says: "Data augmentations... mirrored those used for ResNet-50"
            # So we apply augmentations (on RGB), then normalize, then flip channels?
            # Usually normalization is specific to the model. ResNet uses ImageNet stats.
            # GSViT might use different stats or just raw BGR.
            # "Validation and test images were resized and normalized in the same way but without augmentation."
            # "Input images were preprocessed to match GSViT’s expected BGR format by flipping the first and third channels."
            pass

        if self.transform:
            image = self.transform(image)
            
        if self.is_gsvit:
            # Assuming image is now a Tensor [C, H, W] (RGB)
            # Flip channels to BGR: [0, 1, 2] -> [2, 1, 0]
            if isinstance(image, torch.Tensor):
                image = image[[2, 1, 0], :, :]
        
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(split, is_gsvit=False):
    """
    Returns transforms for training or validation/test.
    """
    # Mean and Std for ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if split == 'train':
        transform_list = [
            transforms.Resize((224, 224)), # Already resized in preprocess, but good to ensure
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    else:
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        
    return transforms.Compose(transform_list)
