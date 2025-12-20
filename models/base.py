import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseClassifier(nn.Module, ABC):
    """Base class for all classifiers."""
    
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def get_features(self, x):
        """Extract features before classifier head."""
        pass
    
    def freeze_backbone(self):
        """Freeze backbone for two-stage training."""
        for name, param in self.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = False
        print("Backbone frozen")
                
    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")
    
    def get_trainable_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
