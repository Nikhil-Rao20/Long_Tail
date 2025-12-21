import torch
import torch.nn as nn
import torchvision.models as models

from .base import BaseClassifier


class ConvNeXtTinyClassifier(BaseClassifier):
    """
    ConvNeXt-Tiny - modern CNN architecture (2022).
    Competitive with Vision Transformers, pure CNN.
    """
    
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super().__init__(num_classes, pretrained)
        
        if pretrained:
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            self.backbone = models.convnext_tiny(weights=weights)
        else:
            self.backbone = models.convnext_tiny(weights=None)
        
        self.feature_dim = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        return self.backbone(x)


class ConvNeXtSmallClassifier(BaseClassifier):
    """ConvNeXt-Small - larger capacity than Tiny."""
    
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super().__init__(num_classes, pretrained)
        
        if pretrained:
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            self.backbone = models.convnext_small(weights=weights)
        else:
            self.backbone = models.convnext_small(weights=None)
        
        self.feature_dim = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        return self.backbone(x)


class ConvNeXtBaseClassifier(BaseClassifier):
    """ConvNeXt-Base - best accuracy, requires more GPU memory."""
    
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super().__init__(num_classes, pretrained)
        
        if pretrained:
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            self.backbone = models.convnext_base(weights=weights)
        else:
            self.backbone = models.convnext_base(weights=None)
        
        self.feature_dim = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        return self.backbone(x)
