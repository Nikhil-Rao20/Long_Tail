# CXR-LT ISBI 2026 - Experiment Tracking

Fill in mAP values as you complete experiments.

---

## Phase 1-2: Backbone × Loss Function

| Backbone | BCE | Focal | CB-Focal | LDAM | LDAM+DRW | Asymmetric |
|----------|-----|-------|----------|------|----------|------------|
| ResNet50 | | | | | | |
| ResNet101 | | | | | | |
| DenseNet121 | | | | | | |
| DenseNet169 | | | | | | |
| EfficientNet-B4 | | | | | | |
| EfficientNet-V2-S | | | | | | |
| ConvNeXt-Tiny | | | | | | |
| ConvNeXt-Small | | | | | | |

---

## Phase 2: Backbone × Sampler (with LDAM+DRW loss)

| Backbone | Random | ClassBalanced | SquareRoot | Progressive |
|----------|--------|---------------|------------|-------------|
| ResNet50 | | | | |
| DenseNet121 | | | | |
| EfficientNet-B4 | | | | |
| ConvNeXt-Tiny | | | | |

---

## Phase 2: Training Strategy Comparison

| Strategy | ResNet50 | DenseNet121 | EfficientNet-B4 |
|----------|----------|-------------|-----------------|
| Single-stage (BCE) | | | |
| Single-stage (LDAM+DRW) | | | |
| Two-stage cRT | | | |

---

## Phase 3: CXR Pretrained Weights

| Model | ImageNet | CXR-Pretrained | TorchXRayVision |
|-------|----------|----------------|-----------------|
| DenseNet121 | | | |
| ResNet50 | | | |

---

## Phase 3: ML-GCN Experiments

| Backbone | Standard Head | ML-GCN | ML-GCN + Attention |
|----------|---------------|--------|-------------------|
| ResNet50 | | | |
| DenseNet121 | | | |

---

## Phase 3-4: Ensemble Combinations

| Ensemble | Averaging | Geometric | Weighted | Rank |
|----------|-----------|-----------|----------|------|
| Best 2 models | | | | |
| Best 3 models | | | | |
| Best 5 models | | | | |

---

## Phase 4: Test-Time Augmentation (TTA)

| Model | No TTA | Flip | Light | Medium |
|-------|--------|------|-------|--------|
| Best Single | | | | |
| Best Ensemble | | | | |

---

## Phase 4: Calibration & Thresholds

| Method | ECE ↓ | mAP | Macro-F1 |
|--------|-------|-----|----------|
| Uncalibrated | | | |
| Temperature Scaling | | | |
| Isotonic | | | |
| Per-class Thresholds | | | |

---

# HOW TO RUN EACH EXPERIMENT

## 📓 Notebook Reference

| Phase | Notebook | Purpose |
|-------|----------|---------|
| 1 | `train_phase1.ipynb` | Baseline training |
| 2 | `train_phase2.ipynb` | Long-tail techniques |
| 3 | `train_phase3.ipynb` | Advanced backbones, ML-GCN, ensemble |
| 4 | `train_phase4.ipynb` | TTA, calibration, thresholds |

---

## 🔧 Phase 1-2: Backbone × Loss Experiments

**Notebook:** `train_phase1.ipynb` or `train_phase2.ipynb`

**To change backbone:**
```python
# In the model creation cell, change MODEL_NAME:
MODEL_NAME = "resnet50"      # Options: resnet50, resnet101, densenet121, 
                             # densenet169, efficientnet_b4, efficientnet_v2_s,
                             # convnext_tiny, convnext_small, convnext_base

model = get_model(MODEL_NAME, num_classes=30, pretrained=True)
```

**To change loss function:**
```python
# In train_phase2.ipynb, modify the criterion:

# BCE (baseline)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Focal Loss
from losses import FocalLoss
criterion = FocalLoss(alpha=1.0, gamma=2.0)

# Class-Balanced Focal
from losses import ClassBalancedFocalLoss
criterion = ClassBalancedFocalLoss(class_counts, beta=0.9999, gamma=2.0)

# LDAM
from losses import LDAMLoss
criterion = LDAMLoss(class_counts, max_m=0.5, s=30)

# LDAM + DRW (recommended)
from losses import LDAMDRWLoss
criterion = LDAMDRWLoss(class_counts, max_m=0.5, s=30, drw_start_epoch=5)

# Asymmetric Loss
from losses import AsymmetricLoss
criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
```

---

## 🔧 Phase 2: Sampler Experiments

**Notebook:** `train_phase2.ipynb`

**To change sampler:**
```python
# Modify create_dataloaders call:

# Random (default)
train_loader, val_loader = create_dataloaders(..., sampler_type=None)

# Class-Balanced
train_loader, val_loader = create_dataloaders(..., sampler_type="class_balanced")

# Square-Root
train_loader, val_loader = create_dataloaders(..., sampler_type="sqrt")

# Progressive
train_loader, val_loader = create_dataloaders(..., sampler_type="progressive")
```

---

## 🔧 Phase 2: Two-Stage Training (cRT)

**Notebook:** `train_phase2.ipynb` → Option 2

```python
from trainer import train_two_stage

history = train_two_stage(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    device=device,
    stage1_epochs=5,    # Freeze backbone, train head
    stage2_epochs=15,   # Fine-tune all
    stage1_lr=1e-3,
    stage2_lr=1e-5,
    save_path="checkpoints/crt_best.pth"
)
```

---

## 🔧 Phase 3: CXR Pretrained Weights

**Notebook:** `train_phase3.ipynb` → Option 2

```python
# With custom pretrained weights (download first):
from models import CXRDenseNet121

model = CXRDenseNet121(
    num_classes=30,
    pretrained_path="path/to/chexpert_weights.pth",
    pretrained_source="chexpert",  # or "mimic", "custom"
    freeze_backbone=False
)

# With TorchXRayVision (pip install torchxrayvision):
from models import TorchXRayVisionWrapper

model = TorchXRayVisionWrapper(
    num_classes=30,
    weights="densenet121-res512-all",
    freeze_backbone=True
)
```

---

## 🔧 Phase 3: ML-GCN Experiments

**Notebook:** `train_phase3.ipynb` → Option 3

```python
from models import MLGCN, MLGCNWithAttention, build_adjacency_matrix

# Build adjacency matrix from training labels
train_labels = train_df[config.CLASS_NAMES].values
adj_matrix = build_adjacency_matrix(train_labels, num_classes=30)

# Standard ML-GCN
model = MLGCN(
    num_classes=30,
    backbone="resnet50",  # or "densenet121"
    adj_matrix=adj_matrix,
    t=0.4  # co-occurrence threshold
)

# ML-GCN with Attention
model = MLGCNWithAttention(
    num_classes=30,
    backbone="resnet50",
    adj_matrix=adj_matrix
)
```

---

## 🔧 Phase 3-4: Ensemble

**Notebook:** `train_phase3.ipynb` → Option 4 or `train_phase4.ipynb`

```python
from ensemble import ensemble_predict, weighted_average_predictions

# Load trained models
models = [model1, model2, model3]  # Your trained models

# Simple averaging
preds = ensemble_predict(models, test_loader, device, weights=None)

# Weighted (based on val mAP)
preds = ensemble_predict(models, test_loader, device, weights=[0.4, 0.35, 0.25])

# With TTA
preds = ensemble_predict(models, test_loader, device, use_tta=True)
```

---

## 🔧 Phase 4: TTA

**Notebook:** `train_phase4.ipynb`

```python
from tta import tta_predict, TTA_CONFIGS

# Available configs: "none", "flip", "light", "medium", "heavy"
predictions = tta_predict(
    model=model,
    dataloader=test_loader,
    device=device,
    tta_config="flip",      # Best speed/accuracy tradeoff
    merge_mode="mean"       # or "max", "gmean"
)
```

---

## 🔧 Phase 4: Calibration

**Notebook:** `train_phase4.ipynb`

```python
from calibration import Calibrator, compare_calibration_methods

# Compare all methods
results = compare_calibration_methods(val_logits, val_probs, val_labels, num_classes=30)

# Use best method
calibrator = Calibrator(num_classes=30, method="temperature")
calibrator.fit(val_logits, val_labels)
calibrated_probs = calibrator.calibrate(test_logits)
```

---

## 🔧 Phase 4: Threshold Optimization

**Notebook:** `train_phase4.ipynb`

```python
from threshold_optimizer import ThresholdOptimizer, evaluate_threshold_strategies

# Compare strategies
results = evaluate_threshold_strategies(val_labels, val_probs, class_counts)

# Use per-class optimal
optimizer = ThresholdOptimizer(num_classes=30, strategy="per_class_f1")
optimizer.fit(val_labels, val_probs)
binary_preds = optimizer.predict(test_probs)
```

---

## 🎯 Recommended Priority Order

| Priority | Experiment | Expected Impact |
|----------|------------|-----------------|
| 1 | ResNet50 + BCE | Baseline |
| 2 | ResNet50 + LDAM+DRW | +2-4% mAP |
| 3 | DenseNet121 + LDAM+DRW | +1-2% mAP |
| 4 | EfficientNet-B4 + Asymmetric | +1-2% mAP |
| 5 | Two-stage cRT | +0.5-1% mAP |
| 6 | Ensemble (top 3) | +1-2% mAP |
| 7 | TTA (flip) | +0.5-1% mAP |
| 8 | ML-GCN | +0.5-1% mAP |
| 9 | CXR Pretrained | +1-3% mAP |

---

## 💾 Saving Results

After each experiment, save:
```python
# Save checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'val_mAP': best_mAP,
    'config': {'backbone': MODEL_NAME, 'loss': LOSS_NAME}
}, f"checkpoints/{MODEL_NAME}_{LOSS_NAME}_mAP{best_mAP:.4f}.pth")
```

Then fill in the mAP value in the table above!