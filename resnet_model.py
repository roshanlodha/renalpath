import torch
import torch.nn as nn
from torchvision import models

def get_resnet_model(num_classes=5, pretrained=True):
    """
    Returns a ResNet-50 model with modified classification head.
    """
    # Load ResNet-50
    # weights='IMAGENET1K_V1' is the modern way, but pretrained=True is legacy compatible
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
    except AttributeError:
        # Fallback for older torchvision versions
        model = models.resnet50(pretrained=pretrained)
    
    # Replace fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
