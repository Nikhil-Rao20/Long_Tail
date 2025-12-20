import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(y_true, y_pred, threshold=0.5):
    """Compute evaluation metrics for multi-label classification."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Mean Average Precision (primary metric)
    try:
        mAP = average_precision_score(y_true, y_pred, average='macro')
    except:
        mAP = 0.0
    
    # Mean AUC
    try:
        aucs = []
        for i in range(y_true.shape[1]):
            if len(np.unique(y_true[:, i])) > 1:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
        mAUC = np.mean(aucs) if aucs else 0.0
    except:
        mAUC = 0.0
    
    # Mean F1 (with threshold)
    try:
        y_pred_binary = (y_pred >= threshold).astype(int)
        f1s = []
        for i in range(y_true.shape[1]):
            if y_true[:, i].sum() > 0:
                f1 = f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
                f1s.append(f1)
        mF1 = np.mean(f1s) if f1s else 0.0
    except:
        mF1 = 0.0
    
    # Per-class AP
    per_class_ap = []
    for i in range(y_true.shape[1]):
        try:
            if len(np.unique(y_true[:, i])) > 1:
                ap = average_precision_score(y_true[:, i], y_pred[:, i])
                per_class_ap.append(ap)
            else:
                per_class_ap.append(0.0)
        except:
            per_class_ap.append(0.0)
    
    return {
        'mAP': mAP,
        'mAUC': mAUC,
        'mF1': mF1,
        'per_class_ap': per_class_ap
    }


def compute_ece(y_true, y_pred, n_bins=15):
    """Compute Expected Calibration Error."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = y_pred[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
            
    return ece
