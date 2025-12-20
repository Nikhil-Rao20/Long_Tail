import torch
import torch.nn as nn


class BCEWithLogitsLossWeighted(nn.Module):
    """BCEWithLogitsLoss with optional positive weights."""
    
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification."""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce_loss
        return loss.mean()


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss (Cui et al., CVPR 2019)."""
    
    def __init__(self, class_counts, beta=0.9999, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        
        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(beta, torch.tensor(class_counts, dtype=torch.float32))
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer('weights', weights)
        
    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        weights = self.weights.to(logits.device)
        cb_weight = targets * weights + (1 - targets)
        
        loss = cb_weight * focal_weight * bce_loss
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (Ridnik et al., ICCV 2021)."""
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Asymmetric clipping
        probs_pos = probs
        probs_neg = probs.clamp(max=1 - self.clip) if self.clip > 0 else probs
        
        # Basic BCE
        loss_pos = targets * torch.log(probs_pos + self.eps)
        loss_neg = (1 - targets) * torch.log(1 - probs_neg + self.eps)
        
        # Asymmetric focusing
        pt_pos = probs_pos
        pt_neg = 1 - probs_neg
        
        weight_pos = (1 - pt_pos) ** self.gamma_pos
        weight_neg = pt_neg ** self.gamma_neg
        
        loss = -weight_pos * loss_pos - weight_neg * loss_neg
        return loss.mean()


def get_loss_function(loss_type, class_counts=None, pos_weight=None):
    """Factory function to get loss function."""
    if loss_type == "bce":
        return BCEWithLogitsLossWeighted(pos_weight=pos_weight)
    elif loss_type == "focal":
        return FocalLoss(gamma=2.0)
    elif loss_type == "cb_focal":
        assert class_counts is not None, "class_counts required for cb_focal"
        return ClassBalancedFocalLoss(class_counts, beta=0.9999, gamma=2.0)
    elif loss_type == "asymmetric":
        return AsymmetricLoss(gamma_neg=4, gamma_pos=1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
