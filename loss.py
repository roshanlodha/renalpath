import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float or list/tensor): Weighting factor for each class. 
                                          If float, acts as a global scalar (e.g., 1.0).
                                          If list/tensor, must match num_classes.
            gamma (float): Focusing parameter. Default 2.0.
            reduction (string): 'mean', 'sum' or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (batch_size, num_classes) - raw logits
        targets: (batch_size) - class indices
        """
        # 1. Calculate CrossEntropy with reduction='none' to get log(pt) per sample
        # We need log_softmax for numeric stability in CE
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 2. Apply Focal Loss formula: -alpha * (1 - pt)^gamma * log(pt)
        # Note: ce_loss is -log(pt) already.
        # So formula becomes: alpha * (1 - pt)^gamma * ce_loss
        
        # Handle alpha
        if isinstance(self.alpha, (list, tuple, torch.Tensor, float, int)):
            if isinstance(self.alpha, (float, int)):
                 alpha_factor = self.alpha
            else:
                # Assuming self.alpha is a tensor of shape [num_classes]
                # We need to pick the alpha corresponding to the target class
                if not isinstance(self.alpha, torch.Tensor):
                    self.alpha = torch.tensor(self.alpha).to(inputs.device)
                alpha_factor = self.alpha[targets]
        else:
            alpha_factor = 1.0
            
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
