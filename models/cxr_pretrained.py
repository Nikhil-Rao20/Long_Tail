"""
CXR Pre-trained Weight Loading Utilities

Supports loading weights from:
- CheXpert pretrained DenseNet121 (Stanford ML Group)
- MIMIC-CXR pretrained models
- TorchXRayVision pretrained models
- Custom pretrained checkpoints

For CXR-LT competition, using CXR-specific pretrained weights 
provides significant improvement over ImageNet pretraining.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from typing import Optional, Dict, Any

from .base import BaseClassifier


# URLs for popular CXR pretrained weights
CXR_PRETRAINED_URLS = {
    # TorchXRayVision DenseNet121 weights (requires torchxrayvision package)
    "torchxrayvision_densenet121_all": "https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt",
    # Alternative: download manually from source repos
}


def load_state_dict_flexible(model: nn.Module, state_dict: Dict[str, Any], strict: bool = False):
    """
    Flexibly load state dict, handling mismatched keys.
    Useful when loading pretrained weights with different classifier heads.
    """
    model_dict = model.state_dict()
    
    # Filter out mismatched keys
    pretrained_dict = {}
    skipped_keys = []
    
    for k, v in state_dict.items():
        # Handle different naming conventions
        new_key = k
        
        # Remove common prefixes
        for prefix in ['module.', 'model.', 'backbone.', 'encoder.']:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        
        if new_key in model_dict:
            if model_dict[new_key].shape == v.shape:
                pretrained_dict[new_key] = v
            else:
                skipped_keys.append(f"{new_key}: shape mismatch {v.shape} vs {model_dict[new_key].shape}")
        else:
            skipped_keys.append(f"{k}: not in model")
    
    if skipped_keys and len(skipped_keys) < 20:
        print(f"Skipped keys: {skipped_keys}")
    elif skipped_keys:
        print(f"Skipped {len(skipped_keys)} keys (showing first 10):")
        for k in skipped_keys[:10]:
            print(f"  {k}")
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    loaded_count = len(pretrained_dict)
    total_count = len(model_dict)
    print(f"Loaded {loaded_count}/{total_count} parameters from pretrained weights")
    
    return model


class CXRDenseNet121(BaseClassifier):
    """
    DenseNet121 with support for CXR-specific pretrained weights.
    
    Supports:
    - ImageNet pretraining (default)
    - CheXpert pretrained weights
    - MIMIC-CXR pretrained weights
    - TorchXRayVision pretrained weights
    - Custom checkpoint path
    """
    
    def __init__(
        self, 
        num_classes: int,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        pretrained_source: str = "imagenet",  # "imagenet", "chexpert", "mimic", "torchxrayvision", "custom"
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        super().__init__(num_classes, pretrained)
        
        self.pretrained_source = pretrained_source
        
        # Initialize backbone
        if pretrained_source == "imagenet" and pretrained:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            self.backbone = models.densenet121(weights=weights)
        else:
            self.backbone = models.densenet121(weights=None)
        
        self.feature_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Load CXR pretrained weights if specified
        if pretrained_path and pretrained_source != "imagenet":
            self._load_cxr_pretrained(pretrained_path, pretrained_source)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        if freeze_backbone:
            self.freeze_backbone()
    
    def _load_cxr_pretrained(self, pretrained_path: str, source: str):
        """Load CXR-specific pretrained weights."""
        print(f"Loading {source} pretrained weights from: {pretrained_path}")
        
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load weights flexibly
        load_state_dict_flexible(self.backbone, state_dict)
        print(f"Successfully loaded {source} pretrained weights")
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        return self.backbone(x)


class CXRResNet50(BaseClassifier):
    """
    ResNet50 with support for CXR-specific pretrained weights.
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        pretrained_source: str = "imagenet",
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        super().__init__(num_classes, pretrained)
        
        self.pretrained_source = pretrained_source
        
        if pretrained_source == "imagenet" and pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        if pretrained_path and pretrained_source != "imagenet":
            self._load_cxr_pretrained(pretrained_path, pretrained_source)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        if freeze_backbone:
            self.freeze_backbone()
    
    def _load_cxr_pretrained(self, pretrained_path: str, source: str):
        """Load CXR-specific pretrained weights."""
        print(f"Loading {source} pretrained weights from: {pretrained_path}")
        
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        load_state_dict_flexible(self.backbone, state_dict)
        print(f"Successfully loaded {source} pretrained weights")
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        return self.backbone(x)


def try_load_torchxrayvision():
    """
    Try to import torchxrayvision for pretrained CXR models.
    Install with: pip install torchxrayvision
    """
    try:
        import torchxrayvision as xrv
        return xrv
    except ImportError:
        print("torchxrayvision not installed. Install with: pip install torchxrayvision")
        return None


class TorchXRayVisionWrapper(BaseClassifier):
    """
    Wrapper for TorchXRayVision pretrained models.
    These models are trained on multiple CXR datasets including:
    - NIH ChestX-ray14
    - CheXpert
    - MIMIC-CXR
    - PadChest
    - Google CXR
    - OpenI
    
    Usage:
        model = TorchXRayVisionWrapper(num_classes=30, weights="densenet121-res512-all")
    """
    
    def __init__(
        self,
        num_classes: int,
        weights: str = "densenet121-res512-all",
        dropout: float = 0.5,
        freeze_backbone: bool = True
    ):
        super().__init__(num_classes, pretrained=True)
        
        xrv = try_load_torchxrayvision()
        if xrv is None:
            raise ImportError("torchxrayvision is required for this model")
        
        # Load pretrained model
        self.backbone = xrv.models.DenseNet(weights=weights)
        
        # Get feature dimension (before the classifier)
        self.feature_dim = self.backbone.classifier.in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        self.backbone.op_threshs = None  # Disable built-in thresholding
        
        # New classifier for our task
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        if freeze_backbone:
            self.freeze_backbone()
            print("Backbone frozen - only classifier will be trained")
    
    def forward(self, x):
        # TorchXRayVision expects normalized images in range [-1024, 1024]
        # We need to transform from [0, 1] range
        # Note: You may need to adjust preprocessing for your data
        features = self.backbone.features(x)
        out = nn.functional.relu(features)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits
    
    def get_features(self, x):
        features = self.backbone.features(x)
        out = nn.functional.relu(features)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)


# Convenience functions
def create_cxr_densenet121(
    num_classes: int,
    pretrained_path: Optional[str] = None,
    pretrained_source: str = "imagenet",
    **kwargs
) -> CXRDenseNet121:
    """Create DenseNet121 with optional CXR pretraining."""
    return CXRDenseNet121(
        num_classes=num_classes,
        pretrained_path=pretrained_path,
        pretrained_source=pretrained_source,
        **kwargs
    )


def create_cxr_resnet50(
    num_classes: int,
    pretrained_path: Optional[str] = None,
    pretrained_source: str = "imagenet",
    **kwargs
) -> CXRResNet50:
    """Create ResNet50 with optional CXR pretraining."""
    return CXRResNet50(
        num_classes=num_classes,
        pretrained_path=pretrained_path,
        pretrained_source=pretrained_source,
        **kwargs
    )
