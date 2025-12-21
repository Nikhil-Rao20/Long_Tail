"""
Threshold Optimization for Multi-Label Classification

In multi-label classification, each class needs its own optimal threshold.
Default threshold of 0.5 is often suboptimal, especially for imbalanced data.

Methods:
1. Fixed threshold (0.5)
2. Per-class optimal threshold (maximize F1)
3. Per-class threshold based on precision-recall tradeoff
4. Global threshold search
5. Class-frequency based thresholds

Note: For mAP evaluation, thresholds don't matter (mAP uses ranking).
But for generating binary predictions (like F1, submission), thresholds matter.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm


def find_optimal_threshold_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_thresholds: int = 100
) -> Tuple[float, float]:
    """
    Find optimal threshold that maximizes F1 score for a single class.
    
    Args:
        y_true: Ground truth labels (binary)
        y_pred: Predicted probabilities
        num_thresholds: Number of thresholds to try
    
    Returns:
        Tuple of (optimal_threshold, best_f1)
    """
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        binary_pred = (y_pred >= thresh).astype(int)
        if binary_pred.sum() == 0:  # All zeros
            continue
        f1 = f1_score(y_true, binary_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1


def find_optimal_thresholds_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    num_thresholds: int = 100,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Find optimal threshold for each class independently.
    
    Args:
        y_true: Ground truth [N, num_classes]
        y_pred: Predictions [N, num_classes]
        class_names: Names of classes (for display)
        num_thresholds: Number of thresholds to try per class
        verbose: Whether to print results
    
    Returns:
        Dictionary mapping class index/name to optimal threshold
    """
    num_classes = y_true.shape[1]
    thresholds = {}
    f1_scores = {}
    
    for c in range(num_classes):
        thresh, f1 = find_optimal_threshold_f1(
            y_true[:, c], 
            y_pred[:, c],
            num_thresholds
        )
        
        class_key = class_names[c] if class_names else c
        thresholds[class_key] = thresh
        f1_scores[class_key] = f1
    
    if verbose:
        print("Optimal thresholds per class:")
        print("-" * 50)
        for i, (key, thresh) in enumerate(thresholds.items()):
            f1 = f1_scores[key]
            print(f"  {key:30s}: {thresh:.3f} (F1={f1:.4f})")
    
    return thresholds


def find_threshold_at_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_precision: float = 0.9
) -> float:
    """
    Find threshold that achieves target precision for a single class.
    Useful when false positives are costly.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Find threshold where precision >= target
    valid_idx = np.where(precision >= target_precision)[0]
    if len(valid_idx) == 0:
        return 0.99  # Very high threshold
    
    # Return threshold with highest recall among valid
    best_idx = valid_idx[np.argmax(recall[valid_idx])]
    if best_idx >= len(thresholds):
        return thresholds[-1]
    return thresholds[best_idx]


def find_threshold_at_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_recall: float = 0.9
) -> float:
    """
    Find threshold that achieves target recall for a single class.
    Useful when false negatives are costly.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Find threshold where recall >= target
    valid_idx = np.where(recall >= target_recall)[0]
    if len(valid_idx) == 0:
        return 0.01  # Very low threshold
    
    # Return threshold with highest precision among valid
    best_idx = valid_idx[np.argmax(precision[valid_idx])]
    if best_idx >= len(thresholds):
        return thresholds[-1]
    return thresholds[best_idx]


def frequency_based_thresholds(
    class_counts: np.ndarray,
    min_threshold: float = 0.1,
    max_threshold: float = 0.9
) -> np.ndarray:
    """
    Set thresholds based on class frequency.
    
    Intuition: Rare classes should have lower thresholds
    (be more permissive) since models tend to underpredict them.
    
    Args:
        class_counts: Number of samples per class
        min_threshold: Minimum threshold (for rarest class)
        max_threshold: Maximum threshold (for most common class)
    
    Returns:
        Array of thresholds per class
    """
    # Normalize to [0, 1]
    log_counts = np.log(class_counts + 1)
    normalized = (log_counts - log_counts.min()) / (log_counts.max() - log_counts.min() + 1e-8)
    
    # Map to threshold range (rare classes get lower thresholds)
    thresholds = min_threshold + normalized * (max_threshold - min_threshold)
    
    return thresholds


def optimize_global_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "macro_f1",
    num_thresholds: int = 100
) -> Tuple[float, float]:
    """
    Find single global threshold that optimizes given metric.
    
    Args:
        y_true: Ground truth [N, num_classes]
        y_pred: Predictions [N, num_classes]
        metric: Metric to optimize ("macro_f1", "micro_f1", "subset_accuracy")
        num_thresholds: Number of thresholds to try
    
    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_score = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        binary_pred = (y_pred >= thresh).astype(int)
        
        if metric == "macro_f1":
            score = f1_score(y_true, binary_pred, average='macro', zero_division=0)
        elif metric == "micro_f1":
            score = f1_score(y_true, binary_pred, average='micro', zero_division=0)
        elif metric == "subset_accuracy":
            score = (binary_pred == y_true).all(axis=1).mean()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def apply_thresholds(
    predictions: np.ndarray,
    thresholds: np.ndarray
) -> np.ndarray:
    """
    Apply per-class thresholds to get binary predictions.
    
    Args:
        predictions: Predicted probabilities [N, num_classes]
        thresholds: Threshold per class [num_classes,] or dict
    
    Returns:
        Binary predictions [N, num_classes]
    """
    if isinstance(thresholds, dict):
        thresholds = np.array(list(thresholds.values()))
    
    return (predictions >= thresholds).astype(int)


class ThresholdOptimizer:
    """
    Class to find and apply optimal thresholds.
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: List[str] = None,
        strategy: str = "per_class_f1"
    ):
        """
        Args:
            num_classes: Number of classes
            class_names: Names of classes
            strategy: Optimization strategy
                - "per_class_f1": Optimize F1 per class
                - "global_f1": Single threshold for all classes
                - "frequency_based": Based on class frequency
                - "precision_target": Target specific precision
                - "recall_target": Target specific recall
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.strategy = strategy
        self.thresholds = np.full(num_classes, 0.5)
        self.is_fitted = False
    
    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_counts: np.ndarray = None,
        target_precision: float = 0.9,
        target_recall: float = 0.9,
        **kwargs
    ) -> 'ThresholdOptimizer':
        """
        Find optimal thresholds from validation data.
        """
        if self.strategy == "per_class_f1":
            thresh_dict = find_optimal_thresholds_per_class(
                y_true, y_pred, self.class_names, verbose=kwargs.get('verbose', True)
            )
            self.thresholds = np.array(list(thresh_dict.values()))
        
        elif self.strategy == "global_f1":
            thresh, score = optimize_global_threshold(y_true, y_pred, "macro_f1")
            self.thresholds = np.full(self.num_classes, thresh)
            print(f"Global threshold: {thresh:.3f} (Macro F1={score:.4f})")
        
        elif self.strategy == "frequency_based":
            if class_counts is None:
                class_counts = y_true.sum(axis=0)
            self.thresholds = frequency_based_thresholds(class_counts)
            print("Frequency-based thresholds set")
        
        elif self.strategy == "precision_target":
            for c in range(self.num_classes):
                self.thresholds[c] = find_threshold_at_precision(
                    y_true[:, c], y_pred[:, c], target_precision
                )
            print(f"Thresholds set for {target_precision:.0%} precision target")
        
        elif self.strategy == "recall_target":
            for c in range(self.num_classes):
                self.thresholds[c] = find_threshold_at_recall(
                    y_true[:, c], y_pred[:, c], target_recall
                )
            print(f"Thresholds set for {target_recall:.0%} recall target")
        
        self.is_fitted = True
        return self
    
    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply thresholds to get binary predictions."""
        return apply_thresholds(y_pred, self.thresholds)
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get thresholds as dictionary."""
        return {name: thresh for name, thresh in zip(self.class_names, self.thresholds)}
    
    def set_thresholds(self, thresholds: np.ndarray):
        """Manually set thresholds."""
        self.thresholds = np.array(thresholds)
        self.is_fitted = True


def evaluate_threshold_strategies(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_counts: np.ndarray = None
) -> Dict[str, Tuple[float, np.ndarray]]:
    """
    Compare different threshold strategies.
    
    Returns:
        Dictionary mapping strategy name to (macro_f1, thresholds)
    """
    results = {}
    num_classes = y_true.shape[1]
    
    # Strategy 1: Default 0.5
    thresholds = np.full(num_classes, 0.5)
    binary_pred = apply_thresholds(y_pred, thresholds)
    f1 = f1_score(y_true, binary_pred, average='macro', zero_division=0)
    results["default_0.5"] = (f1, thresholds.copy())
    
    # Strategy 2: Global optimal
    thresh, _ = optimize_global_threshold(y_true, y_pred, "macro_f1")
    thresholds = np.full(num_classes, thresh)
    binary_pred = apply_thresholds(y_pred, thresholds)
    f1 = f1_score(y_true, binary_pred, average='macro', zero_division=0)
    results["global_optimal"] = (f1, thresholds.copy())
    
    # Strategy 3: Per-class optimal
    thresh_dict = find_optimal_thresholds_per_class(y_true, y_pred, verbose=False)
    thresholds = np.array(list(thresh_dict.values()))
    binary_pred = apply_thresholds(y_pred, thresholds)
    f1 = f1_score(y_true, binary_pred, average='macro', zero_division=0)
    results["per_class_optimal"] = (f1, thresholds.copy())
    
    # Strategy 4: Frequency-based
    if class_counts is None:
        class_counts = y_true.sum(axis=0)
    thresholds = frequency_based_thresholds(class_counts)
    binary_pred = apply_thresholds(y_pred, thresholds)
    f1 = f1_score(y_true, binary_pred, average='macro', zero_division=0)
    results["frequency_based"] = (f1, thresholds.copy())
    
    # Print comparison
    print("\nThreshold Strategy Comparison:")
    print("-" * 40)
    for name, (f1, _) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        print(f"  {name:25s}: Macro F1 = {f1:.4f}")
    
    return results
