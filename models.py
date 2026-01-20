import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import sys

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

        # Torchvision's ViT implementation does not expose `forward_features` like timm.
        # To keep feature extraction robust across versions, capture the input to the
        # classifier head via a forward hook.
        features_list = []

        def hook_fn(module, input, output):
            if input:
                features_list.append(input[0])

        head = None
        for attr in ["heads", "head", "fc", "classifier"]:
            if hasattr(self.model, attr):
                candidate = getattr(self.model, attr)
                if isinstance(candidate, nn.Module):
                    head = candidate
                    break

        if head is None:
            # Best-effort fallback for non-torchvision ViT variants.
            if hasattr(self.model, "forward_features"):
                features = self.model.forward_features(x)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                if hasattr(features, "dim") and features.dim() == 3:
                    features = features[:, 0]
                logits = self.model(x)
                return logits, features

            raise RuntimeError("ViT model does not expose a recognizable head; cannot return features.")

        handle = head.register_forward_hook(hook_fn)
        try:
            logits = self.model(x)
        finally:
            handle.remove()

        features = features_list[0] if features_list else None
        return logits, features


class GSViT_Classifier(nn.Module):
    def __init__(self, model_path, num_classes=5):
        super(GSViT_Classifier, self).__init__()
        self.num_classes = num_classes
        
        checkpoint = self._load_checkpoint(model_path)

        # Case 1: full pickled model object
        if isinstance(checkpoint, nn.Module):
            self.model = checkpoint
            self._replace_head_if_needed(num_classes)
            self._freeze_all()
            self._unfreeze_head()
            return

        # Case 2: state_dict (e.g., EfficientViT autoencoder weights with "evit.*" keys)
        if isinstance(checkpoint, dict):
            state_dict = self._extract_state_dict(checkpoint, model_path)
            self.model = self._build_efficientvit_m5(num_classes=1000)
            mapped_state_dict = self._map_checkpoint_state_dict(state_dict)
            try:
                self.model.load_state_dict(mapped_state_dict, strict=False)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Error loading weights from '{model_path}'. "
                    "Ensure the checkpoint matches the EfficientViT-M5 backbone."
                ) from e

            self._replace_head_if_needed(num_classes)
            self._freeze_all()
            self._unfreeze_head()
            return

        raise TypeError(f"Unsupported GSViT checkpoint type: {type(checkpoint)}")

    def _load_checkpoint(self, model_path):
        try:
            return torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Error loading GSViT checkpoint from '{model_path}': {e}") from e

    def _extract_state_dict(self, checkpoint, model_path):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]

        # Plain state_dict (OrderedDict[str, Tensor])
        if all(isinstance(k, str) for k in checkpoint.keys()) and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

        raise ValueError(
            f"Unsupported checkpoint format in '{model_path}'. Expected a pickled nn.Module or a state_dict-like dict."
        )

    def _ensure_efficientvit_on_path(self):
        repo_root = Path(__file__).resolve().parent
        models_dir = repo_root / "models"
        if models_dir.is_dir():
            models_dir_str = str(models_dir)
            if models_dir_str not in sys.path:
                sys.path.insert(0, models_dir_str)

    def _build_efficientvit_m5(self, num_classes=1000):
        self._ensure_efficientvit_on_path()
        try:
            from EfficientViT.classification.model.efficientvit import EfficientViT
        except Exception as e:
            raise RuntimeError(
                "Could not import EfficientViT. Expected code at 'models/EfficientViT'. "
                "Ensure you have downloaded the GSViT repo's EfficientViT folder into 'models/EfficientViT'."
            ) from e

        model_cfg = {
            "img_size": 224,
            "patch_size": 16,
            "embed_dim": [192, 288, 384],
            "depth": [1, 3, 4],
            "num_heads": [3, 3, 4],
            "window_size": [7, 7, 7],
            "kernels": [7, 5, 3, 3],
        }
        return EfficientViT(num_classes=num_classes, distillation=False, **model_cfg)

    def _map_checkpoint_state_dict(self, state_dict):
        # Support checkpoints saved from DataParallel ("module.*")
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            cleaned[k] = v

        keys = list(cleaned.keys())

        # EfficientViT autoencoder format from GSViT_old.py: "evit.{0..3}.*" + "decoder.*"
        if any(k.startswith("evit.") for k in keys):
            mapped = {}
            for k, v in cleaned.items():
                if not k.startswith("evit."):
                    continue
                rest = k[len("evit."):]
                if rest.startswith("0."):
                    mapped["patch_embed." + rest[2:]] = v
                elif rest.startswith("1."):
                    mapped["blocks1." + rest[2:]] = v
                elif rest.startswith("2."):
                    mapped["blocks2." + rest[2:]] = v
                elif rest.startswith("3."):
                    mapped["blocks3." + rest[2:]] = v
            return mapped

        # Checkpoints saved from this project's GSViT_Classifier: "model.*"
        if any(k.startswith("model.") for k in keys):
            return {k[len("model."):]: v for k, v in cleaned.items() if k.startswith("model.")}

        # Plain EfficientViT state_dict
        return cleaned

    def _freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_head(self):
        for name in ["head", "fc", "classifier", "heads"]:
            if hasattr(self.model, name):
                module = getattr(self.model, name)
                if isinstance(module, nn.Module):
                    for param in module.parameters():
                        param.requires_grad = True
                return

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
