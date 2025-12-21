import torch
import torch.nn as nn
import torchvision.models as models

from .base import BaseClassifier


class DenseNet121Classifier(BaseClassifier):
    """
    DenseNet121 backbone - standard for CXR classification.
    Used in CheXNet and many medical imaging papers.
    """
    
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super().__init__(num_classes, pretrained)
        
        if pretrained:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            self.backbone = models.densenet121(weights=weights)
        else:
            self.backbone = models.densenet121(weights=None)
        
        self.feature_dim = self.backbone.classifier.in_features
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


class DenseNet169Classifier(BaseClassifier):
    """DenseNet169 - deeper variant with more capacity."""
    
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super().__init__(num_classes, pretrained)
        
        if pretrained:
            weights = models.DenseNet169_Weights.IMAGENET1K_V1
            self.backbone = models.densenet169(weights=weights)
        else:
            self.backbone = models.densenet169(weights=None)
        
        self.feature_dim = self.backbone.classifier.in_features
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
