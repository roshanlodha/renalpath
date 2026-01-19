import torch
import torch.nn as nn
from torchvision import models

class ResNet50_Classifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNet50_Classifier, self).__init__()
        
        # Load backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Replace fc layer with custom head
        # Original fc is Linear(2048, 1000)
        in_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class GSViT_Classifier(nn.Module):
    def __init__(self, model_path, num_classes=5):
        super(GSViT_Classifier, self).__init__()
        
        # Load the pre-trained model
        try:
            # weights_only=False is required for loading full model objects (pickled classes)
            # The prompt implies loading .pkl generic objects.
            self.model = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load GSViT model from {model_path}: {e}")
            
        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False
            
        # We need to ensure the head handles num_classes.
        # Assuming the loaded model is a full model.
        # Check if we need to replace the head or if it's already compatible.
        # The prompt implies we might use this as a feature extractor or finetuner.
        # "Load pre-trained .pkl... Unfreezing Strategy... provide method unfreeze_last_blocks"
        # It DOES NOT explicitly say to replace the head like ResNet, but usually we must if classes differ.
        # However, Phase 1 analysis showed gsvit_model.py replacing the head.
        # Let's inspect the model structure dynamically if possible, or assume generic "head" or "fc".
        
        # IMPORTANT: The prompt for Phase 2 only specified: "Load pre-trained... Unfreezing Strategy...".
        # It did NOT specify replacing the head for GSViT. However, it's a "Classifier", so it likely needs one.
        # I will assume we should just allow the loaded model to be used, but provide unfreeze logic.
        # But if the loaded model has 1000 classes and we need 5, we MUST replace the head.
        # I will add logic to try and replace the head if a common name is found, consistent with typical transfer learning.
        
        self._replace_head_if_needed(num_classes)


    def _replace_head_if_needed(self, num_classes):
        # Heuristic to find and replace head
        possible_head_names = ['head', 'fc', 'classifier', 'layers.head']
        
        replaced = False
        for name in possible_head_names:
            if hasattr(self.model, name):
                module = getattr(self.model, name)
                if isinstance(module, nn.Linear):
                    # Replace
                    in_features = module.in_features
                    new_head = nn.Linear(in_features, num_classes)
                    setattr(self.model, name, new_head)
                    replaced = True
                    break
                elif isinstance(module, nn.Sequential):
                    # Check last layer of sequential
                     if isinstance(module[-1], nn.Linear):
                        in_features = module[-1].in_features
                        # Replace just the last linear, or the whole sequential?
                        # Let's replace the last linear
                        module[-1] = nn.Linear(in_features, num_classes)
                        replaced = True
                        break
        
        if not replaced:
            print("Warning: Could not automatically find and replace classifier head for GSViT. Ensure loaded model matches num_classes.")

    def forward(self, x):
        return self.model(x)

    def unfreeze_last_blocks(self, n=2):
        """
        Unfreezes the last n blocks of the transformer.
        Assumes standard naming conventions for ViT blocks (e.g., 'blocks', 'layers').
        """
        # 1. Unfreeze Head (always learn the classifier)
        possible_head_names = ['head', 'fc', 'classifier']
        for name in possible_head_names:
             if hasattr(self.model, name):
                for param in getattr(self.model, name).parameters():
                    param.requires_grad = True

        # 2. Unfreeze last n blocks
        # We need to find the container of blocks.
        # Common structures: mode.blocks, model.transformer.layers, etc.
        
        block_container = None
        block_container_name = None
        
        # Search for container
        candidates = ['blocks', 'layers', 'features']
        
        # Breadth-first search for a container that is a ModuleList or Sequential having many children
        queue = [(self.model, 'model')]
        
        found_blocks = []
        
        # Heuristic: Find the longest ModuleList/Sequential in the model, likely the blocks
        max_len = 0
        target_module = None
        
        for name, module in self.model.named_modules():
             # Avoid going too deep if we want top-level blocks
             # But named_modules recursively returns everything.
             # We want the container.
             if isinstance(module, (nn.ModuleList, nn.Sequential)):
                 if len(module) > max_len:
                     max_len = len(module)
                     target_module = module
        
        if target_module and max_len >= n:
            # Unfreeze last n modules in this container
            total_blocks = len(target_module)
            start_idx = total_blocks - n
            
            print(f"Unfreezing last {n} blocks (indices {start_idx} to {total_blocks-1}) in identified container.")
            
            for i in range(start_idx, total_blocks):
                for param in target_module[i].parameters():
                    param.requires_grad = True
        else:
             print(f"Warning: Could not identify a block container deep enough to unfreeze {n} blocks.")

