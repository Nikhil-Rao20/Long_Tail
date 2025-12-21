from .base import BaseClassifier
from .resnet import ResNet50Classifier, ResNet101Classifier
from .densenet import DenseNet121Classifier, DenseNet169Classifier
from .efficientnet import EfficientNetB4Classifier, EfficientNetB5Classifier, EfficientNetV2SClassifier
from .convnext import ConvNeXtTinyClassifier, ConvNeXtSmallClassifier, ConvNeXtBaseClassifier
from .cxr_pretrained import CXRDenseNet121, CXRResNet50, TorchXRayVisionWrapper, create_cxr_densenet121, create_cxr_resnet50
from .mlgcn import MLGCN, MLGCNWithAttention, build_adjacency_matrix, build_adjacency_matrix_symmetric

__all__ = [
    "BaseClassifier",
    # ResNet
    "ResNet50Classifier",
    "ResNet101Classifier",
    # DenseNet  
    "DenseNet121Classifier",
    "DenseNet169Classifier",
    # EfficientNet
    "EfficientNetB4Classifier",
    "EfficientNetB5Classifier", 
    "EfficientNetV2SClassifier",
    # ConvNeXt
    "ConvNeXtTinyClassifier",
    "ConvNeXtSmallClassifier",
    "ConvNeXtBaseClassifier",
    # CXR Pretrained
    "CXRDenseNet121",
    "CXRResNet50",
    "TorchXRayVisionWrapper",
    "create_cxr_densenet121",
    "create_cxr_resnet50",
    # ML-GCN
    "MLGCN",
    "MLGCNWithAttention",
    "build_adjacency_matrix",
    "build_adjacency_matrix_symmetric",
]


# Model registry for easy selection
MODEL_REGISTRY = {
    "resnet50": ResNet50Classifier,
    "resnet101": ResNet101Classifier,
    "densenet121": DenseNet121Classifier,
    "densenet169": DenseNet169Classifier,
    "efficientnet_b4": EfficientNetB4Classifier,
    "efficientnet_b5": EfficientNetB5Classifier,
    "efficientnet_v2_s": EfficientNetV2SClassifier,
    "convnext_tiny": ConvNeXtTinyClassifier,
    "convnext_small": ConvNeXtSmallClassifier,
    "convnext_base": ConvNeXtBaseClassifier,
    "cxr_densenet121": CXRDenseNet121,
    "cxr_resnet50": CXRResNet50,
    "mlgcn": MLGCN,
    "mlgcn_attention": MLGCNWithAttention,
}


def get_model(model_name, num_classes, pretrained=True, **kwargs):
    """Get model by name from registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](num_classes=num_classes, pretrained=pretrained, **kwargs)
