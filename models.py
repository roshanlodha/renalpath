import torch
import torch.nn as nn
from torchvision import models

class ResNet50_Classifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNet50_Classifier, self).__init__()
        
        # Load backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Replace fc layer with Identity to get features
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Custom Head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits


class ViT_Classifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, freeze_backbone=True):
        super(ViT_Classifier, self).__init__()

        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.vit_b_16(weights=weights)

        if hasattr(self.model, "heads") and hasattr(self.model.heads, "head") and isinstance(self.model.heads.head, nn.Linear):
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Unexpected ViT head structure; cannot replace classifier head.")

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.heads.parameters():
                param.requires_grad = True

    def forward(self, x, return_features=False):
        if not return_features:
            return self.model(x)

        if not hasattr(self.model, "forward_features"):
            raise RuntimeError("ViT model does not expose forward_features; cannot return features.")

        features = self.model.forward_features(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if hasattr(features, "dim") and features.dim() == 3:
            features = features[:, 0]

        logits = self.model.heads(features)
        return logits, features


class GSViT_Classifier(nn.Module):
    def __init__(self, model_path, num_classes=5):
        super(GSViT_Classifier, self).__init__()
        self.num_classes = num_classes
        
        # Load the pre-trained model (no ViT fallback; ViT is a separate model_type)
        try:
            self.model = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Error loading GSViT model from '{model_path}': {e}") from e

        if isinstance(self.model, dict):
            raise ValueError(
                f"The file '{model_path}' appears to contain a state_dict, but the GSViT model architecture is missing. "
                "Provide a pickled GSViT model object instead."
            )

            
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

    def _get_module_by_path(self, root, path):
        obj = root
        for part in path.split('.'):
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj

    def _replace_head_if_needed(self, num_classes):
        # Heuristic to find and replace head
        possible_head_names = ['heads.head', 'head', 'fc', 'classifier', 'layers.head']
        
        replaced = False
        for name in possible_head_names:
            module = self._get_module_by_path(self.model, name)
            if module is None:
                continue

            if isinstance(module, nn.Linear):
                in_features = module.in_features
                new_head = nn.Linear(in_features, num_classes)
                parent_path, attr = name.rsplit('.', 1) if '.' in name else (None, name)
                if parent_path is None:
                    setattr(self.model, attr, new_head)
                else:
                    parent = self._get_module_by_path(self.model, parent_path)
                    if parent is not None:
                        setattr(parent, attr, new_head)
                replaced = True
                break

            if isinstance(module, nn.Sequential) and len(module) > 0 and isinstance(module[-1], nn.Linear):
                in_features = module[-1].in_features
                module[-1] = nn.Linear(in_features, num_classes)
                replaced = True
                break

        if not replaced:
            print("Warning: Could not automatically find and replace classifier head for GSViT. Ensure loaded model matches num_classes.")

    def _find_head_module_for_features(self):
        candidate_paths = ['heads', 'heads.head', 'head', 'fc', 'classifier', 'layers.head']
        for path in candidate_paths:
            module = self._get_module_by_path(self.model, path)
            if module is None:
                continue
            if isinstance(module, (nn.Linear, nn.Sequential)):
                return module

        matching_linears = []
        all_linears = []
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                all_linears.append(module)
                if module.out_features == self.num_classes:
                    matching_linears.append(module)

        if matching_linears:
            return matching_linears[-1]
        if all_linears:
            return all_linears[-1]
        return None

    def forward(self, x, return_features=False):
        if not return_features:
            return self.model(x)
        
        # Capture features logic
        features_list = []
        def hook_fn(module, input, output):
            # Input to head is the feature vector
            features_list.append(input[0])
            
        # Attach hook to head
        head = self._find_head_module_for_features()
        
        handle = None
        if head is not None:
            handle = head.register_forward_hook(hook_fn)
            
        logits = self.model(x)
        
        if handle:
            handle.remove()
            
        features = features_list[0] if features_list else None
        return logits, features

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
