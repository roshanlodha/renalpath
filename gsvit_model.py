import torch
import torch.nn as nn
import sys
import os

class GSViTClassifier(nn.Module):
    def __init__(self, model_path, num_classes=5):
        super(GSViTClassifier, self).__init__()
        
        # Load the pretrained GSViT backbone
        # Assuming model_path points to a pickled model object or state_dict
        # If it's a state_dict, we need the model class definition which we don't have.
        # We'll assume it's a full model object or TorchScript.
        try:
            self.backbone = torch.load(model_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading GSViT model from {model_path}: {e}")
            raise e
            
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Identify the input dimension for the head
        # We need to find the existing head and get its input features.
        # Common names: head, fc, classifier, layers[-1]
        self.in_features = None
        
        # Attempt to find the head
        possible_heads = ['head', 'fc', 'classifier']
        for name in possible_heads:
            if hasattr(self.backbone, name):
                module = getattr(self.backbone, name)
                if isinstance(module, nn.Linear):
                    self.in_features = module.in_features
                    # Replace with Identity or remove to just get features
                    setattr(self.backbone, name, nn.Identity())
                    break
        
        if self.in_features is None:
            # Fallback: try to pass a dummy input and check output shape
            # This requires knowing the input size expected by GSViT.
            # Assuming 224x224 input.
            print("Could not identify head automatically. Attempting to infer from forward pass...")
            try:
                dummy_input = torch.randn(1, 3, 224, 224)
                # GSViT might expect BGR, but shape is what matters for feature dim
                with torch.no_grad():
                    features = self.backbone(dummy_input)
                self.in_features = features.shape[1]
                print(f"Inferred input features: {self.in_features}")
            except Exception as e:
                print(f"Failed to infer input features: {e}")
                # Default fallback if all else fails, though this is risky
                self.in_features = 1024 # Common for ViTs
        
        # Define new classification head
        # "two hidden layers with 2048 and 512 neurons... ELU... BN... Dropout(0.1)"
        self.classification_head = nn.Sequential(
            nn.Linear(self.in_features, 2048),
            nn.ELU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.1),
            nn.Linear(2048, 512),
            nn.ELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Backbone forward
        features = self.backbone(x)
        
        # If backbone returns a tuple/list (common in transformers), take the first element or CLS token
        if isinstance(features, (tuple, list)):
            features = features[0]
            
        # Flatten if needed (though usually global pool is done in backbone)
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)
            
        return self.classification_head(features)

def get_gsvit_model(model_path='GSViT.pkl', num_classes=5):
    return GSViTClassifier(model_path, num_classes)
