import torch
import torch.nn as nn
import torchvision.models as models

from .base import BaseClassifier


class EfficientNetB4Classifier(BaseClassifier):
    """
    EfficientNet-B4 - good balance of accuracy and efficiency.
    Better than ResNet50 with similar compute.
    """
    
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super().__init__(num_classes, pretrained)
        
        if pretrained:
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b4(weights=weights)
        else:
            self.backbone = models.efficientnet_b4(weights=None)
        
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        return self.backbone(x)


class EfficientNetB5Classifier(BaseClassifier):
    """EfficientNet-B5 - higher capacity, needs more memory."""
    
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super().__init__(num_classes, pretrained)
        
        if pretrained:
            weights = models.EfficientNet_B5_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b5(weights=weights)
        else:
            self.backbone = models.efficientnet_b5(weights=None)
        
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        return self.backbone(x)


class EfficientNetV2SClassifier(BaseClassifier):
    """EfficientNetV2-S - newer architecture, faster training."""
    
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super().__init__(num_classes, pretrained)
        
        if pretrained:
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_v2_s(weights=weights)
        else:
            self.backbone = models.efficientnet_v2_s(weights=None)
        
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        return self.backbone(x)
