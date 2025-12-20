import torch
import torch.nn as nn
import torchvision.models as models

from .base import BaseClassifier


class ResNet50Classifier(BaseClassifier):
    """ResNet50 backbone for multi-label classification."""
    
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super().__init__(num_classes, pretrained)
        
        # Load pretrained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get feature dimension
        self.feature_dim = self.backbone.fc.in_features
        
        # Remove original fc layer
        self.backbone.fc = nn.Identity()
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features before classifier head."""
        return self.backbone(x)


class ResNet101Classifier(BaseClassifier):
    """ResNet101 backbone for multi-label classification."""
    
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super().__init__(num_classes, pretrained)
        
        if pretrained:
            weights = models.ResNet101_Weights.IMAGENET1K_V2
            self.backbone = models.resnet101(weights=weights)
        else:
            self.backbone = models.resnet101(weights=None)
        
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
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
