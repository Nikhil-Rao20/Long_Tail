"""
Probability Calibration for Multi-Label Classification

Neural networks often produce poorly calibrated probabilities:
- Overconfident on wrong predictions
- Underconfident on correct predictions

Calibration ensures P(correct | confidence=p) ≈ p

Methods:
1. Temperature Scaling (single parameter, fast)
2. Platt Scaling (per-class logistic regression)
3. Isotonic Regression (non-parametric, per-class)
4. Beta Calibration (two-parameter)
5. Histogram Binning (simple, per-class)

For mAP, calibration helps if it better separates positives from negatives.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from scipy.optimize import minimize
from tqdm.auto import tqdm


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling - simplest calibration method.
    
    Divides logits by a learned temperature T before softmax/sigmoid.
    T > 1: makes predictions less confident (softer)
    T < 1: makes predictions more confident (sharper)
    
    For multi-label, we use sigmoid after temperature scaling.
    """
    
    def __init__(self, num_classes: int = 1):
        super().__init__()
        # Single temperature for all classes (simpler, more robust)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling."""
        return logits / self.temperature
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> 'TemperatureScaling':
        """
        Find optimal temperature using NLL loss.
        
        Args:
            logits: Model logits [N, num_classes]
            labels: Ground truth [N, num_classes]
            lr: Learning rate
            max_iter: Maximum iterations
        
        Returns:
            self
        """
        logits_tensor = torch.FloatTensor(logits)
        labels_tensor = torch.FloatTensor(labels)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_tensor)
            loss = criterion(scaled_logits, labels_tensor)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply calibration to logits."""
        with torch.no_grad():
            logits_tensor = torch.FloatTensor(logits)
            scaled = self.forward(logits_tensor)
            probs = torch.sigmoid(scaled)
            return probs.numpy()


class PerClassTemperatureScaling(nn.Module):
    """
    Per-class temperature scaling.
    
    Each class gets its own temperature parameter.
    More flexible but may overfit with limited data.
    """
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(num_classes) * 1.5)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperatures
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> 'PerClassTemperatureScaling':
        """Find optimal temperatures per class."""
        logits_tensor = torch.FloatTensor(logits)
        labels_tensor = torch.FloatTensor(labels)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam([self.temperatures], lr=lr)
        
        for _ in range(max_iter):
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_tensor)
            loss = criterion(scaled_logits, labels_tensor)
            loss.backward()
            optimizer.step()
            
            # Keep temperatures positive
            with torch.no_grad():
                self.temperatures.clamp_(min=0.1, max=10.0)
        
        print(f"Temperature range: [{self.temperatures.min().item():.3f}, {self.temperatures.max().item():.3f}]")
        return self
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits_tensor = torch.FloatTensor(logits)
            scaled = self.forward(logits_tensor)
            probs = torch.sigmoid(scaled)
            return probs.numpy()


class PlattScaling:
    """
    Platt Scaling - fits a logistic regression to map scores to probabilities.
    
    For each class: P(y=1|score) = sigmoid(a * score + b)
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.calibrators = [LogisticRegression() for _ in range(num_classes)]
        self.is_fitted = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> 'PlattScaling':
        """Fit logistic regression for each class."""
        for c in range(self.num_classes):
            # Need both positive and negative samples
            if labels[:, c].sum() > 0 and labels[:, c].sum() < len(labels):
                self.calibrators[c].fit(logits[:, c].reshape(-1, 1), labels[:, c])
            else:
                # If only one class, use identity
                self.calibrators[c] = None
        
        self.is_fitted = True
        print("Platt scaling fitted for all classes")
        return self
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to get calibrated probabilities."""
        probs = np.zeros_like(logits)
        
        for c in range(self.num_classes):
            if self.calibrators[c] is not None:
                probs[:, c] = self.calibrators[c].predict_proba(
                    logits[:, c].reshape(-1, 1)
                )[:, 1]
            else:
                probs[:, c] = 1 / (1 + np.exp(-logits[:, c]))  # Standard sigmoid
        
        return probs


class IsotonicCalibration:
    """
    Isotonic Regression calibration.
    
    Non-parametric method that fits a monotonically increasing function.
    Good when the relationship between scores and probabilities is complex.
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.calibrators = [IsotonicRegression(out_of_bounds='clip') for _ in range(num_classes)]
        self.is_fitted = False
    
    def fit(
        self,
        predictions: np.ndarray,  # Already probabilities, not logits
        labels: np.ndarray
    ) -> 'IsotonicCalibration':
        """Fit isotonic regression for each class."""
        for c in range(self.num_classes):
            if labels[:, c].sum() > 0 and labels[:, c].sum() < len(labels):
                self.calibrators[c].fit(predictions[:, c], labels[:, c])
            else:
                self.calibrators[c] = None
        
        self.is_fitted = True
        print("Isotonic calibration fitted for all classes")
        return self
    
    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        calibrated = np.zeros_like(predictions)
        
        for c in range(self.num_classes):
            if self.calibrators[c] is not None:
                calibrated[:, c] = self.calibrators[c].predict(predictions[:, c])
            else:
                calibrated[:, c] = predictions[:, c]
        
        return np.clip(calibrated, 0, 1)


class HistogramBinning:
    """
    Histogram Binning calibration.
    
    Simple method that groups predictions into bins and uses
    the average true probability in each bin.
    """
    
    def __init__(self, num_classes: int, num_bins: int = 15):
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.bin_edges = None
        self.bin_values = None
        self.is_fitted = False
    
    def fit(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> 'HistogramBinning':
        """Fit histogram binning for each class."""
        self.bin_edges = []
        self.bin_values = []
        
        for c in range(self.num_classes):
            # Create uniform bins
            edges = np.linspace(0, 1, self.num_bins + 1)
            values = np.zeros(self.num_bins)
            
            for i in range(self.num_bins):
                mask = (predictions[:, c] >= edges[i]) & (predictions[:, c] < edges[i+1])
                if mask.sum() > 0:
                    values[i] = labels[mask, c].mean()
                else:
                    values[i] = (edges[i] + edges[i+1]) / 2  # Default to bin center
            
            self.bin_edges.append(edges)
            self.bin_values.append(values)
        
        self.is_fitted = True
        print(f"Histogram binning fitted with {self.num_bins} bins")
        return self
    
    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """Apply histogram binning calibration."""
        calibrated = np.zeros_like(predictions)
        
        for c in range(self.num_classes):
            bin_idx = np.digitize(predictions[:, c], self.bin_edges[c]) - 1
            bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
            calibrated[:, c] = self.bin_values[c][bin_idx]
        
        return calibrated


def compute_ece(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = sum(|accuracy - confidence|) weighted by bin size
    
    Lower is better. 0 = perfectly calibrated.
    """
    bin_edges = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = 0
    
    # Flatten for overall ECE
    preds_flat = predictions.flatten()
    labels_flat = labels.flatten()
    
    for i in range(num_bins):
        mask = (preds_flat >= bin_edges[i]) & (preds_flat < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        
        bin_confidence = preds_flat[mask].mean()
        bin_accuracy = labels_flat[mask].mean()
        bin_size = mask.sum()
        
        ece += np.abs(bin_accuracy - bin_confidence) * bin_size
        total_samples += bin_size
    
    return ece / total_samples if total_samples > 0 else 0.0


def compute_mce(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE = max(|accuracy - confidence|) across bins
    """
    bin_edges = np.linspace(0, 1, num_bins + 1)
    mce = 0.0
    
    preds_flat = predictions.flatten()
    labels_flat = labels.flatten()
    
    for i in range(num_bins):
        mask = (preds_flat >= bin_edges[i]) & (preds_flat < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        
        bin_confidence = preds_flat[mask].mean()
        bin_accuracy = labels_flat[mask].mean()
        
        mce = max(mce, np.abs(bin_accuracy - bin_confidence))
    
    return mce


class Calibrator:
    """
    Unified calibrator that wraps different methods.
    """
    
    def __init__(
        self,
        num_classes: int,
        method: str = "temperature"
    ):
        """
        Args:
            num_classes: Number of classes
            method: Calibration method
                - "temperature": Global temperature scaling
                - "temperature_per_class": Per-class temperature
                - "platt": Platt scaling (logistic regression)
                - "isotonic": Isotonic regression
                - "histogram": Histogram binning
        """
        self.num_classes = num_classes
        self.method = method
        
        if method == "temperature":
            self.calibrator = TemperatureScaling(num_classes)
        elif method == "temperature_per_class":
            self.calibrator = PerClassTemperatureScaling(num_classes)
        elif method == "platt":
            self.calibrator = PlattScaling(num_classes)
        elif method == "isotonic":
            self.calibrator = IsotonicCalibration(num_classes)
        elif method == "histogram":
            self.calibrator = HistogramBinning(num_classes)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.is_fitted = False
    
    def fit(
        self,
        logits_or_probs: np.ndarray,
        labels: np.ndarray
    ) -> 'Calibrator':
        """
        Fit calibrator on validation data.
        
        Note: Temperature/Platt scaling expect logits.
              Isotonic/Histogram expect probabilities.
        """
        if self.method in ["temperature", "temperature_per_class", "platt"]:
            # These work with logits
            self.calibrator.fit(logits_or_probs, labels)
        else:
            # These work with probabilities
            self.calibrator.fit(logits_or_probs, labels)
        
        self.is_fitted = True
        return self
    
    def calibrate(self, logits_or_probs: np.ndarray) -> np.ndarray:
        """Apply calibration."""
        return self.calibrator.calibrate(logits_or_probs)
    
    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate calibration quality."""
        ece = compute_ece(predictions, labels)
        mce = compute_mce(predictions, labels)
        
        return {
            "ECE": ece,
            "MCE": mce
        }


def compare_calibration_methods(
    logits: np.ndarray,
    predictions: np.ndarray,  # sigmoid(logits)
    labels: np.ndarray,
    num_classes: int
) -> Dict[str, Dict]:
    """
    Compare different calibration methods.
    
    Returns dictionary with ECE, MCE for each method.
    """
    results = {}
    
    # Uncalibrated
    ece = compute_ece(predictions, labels)
    mce = compute_mce(predictions, labels)
    results["uncalibrated"] = {"ECE": ece, "MCE": mce, "probs": predictions}
    
    # Temperature scaling
    ts = TemperatureScaling(num_classes)
    ts.fit(logits, labels)
    cal_probs = ts.calibrate(logits)
    results["temperature"] = {
        "ECE": compute_ece(cal_probs, labels),
        "MCE": compute_mce(cal_probs, labels),
        "probs": cal_probs,
        "temperature": ts.temperature.item()
    }
    
    # Isotonic
    iso = IsotonicCalibration(num_classes)
    iso.fit(predictions, labels)
    cal_probs = iso.calibrate(predictions)
    results["isotonic"] = {
        "ECE": compute_ece(cal_probs, labels),
        "MCE": compute_mce(cal_probs, labels),
        "probs": cal_probs
    }
    
    # Histogram binning
    hist = HistogramBinning(num_classes)
    hist.fit(predictions, labels)
    cal_probs = hist.calibrate(predictions)
    results["histogram"] = {
        "ECE": compute_ece(cal_probs, labels),
        "MCE": compute_mce(cal_probs, labels),
        "probs": cal_probs
    }
    
    # Print comparison
    print("\nCalibration Method Comparison:")
    print("-" * 50)
    print(f"{'Method':<20} {'ECE':>10} {'MCE':>10}")
    print("-" * 50)
    for method, metrics in sorted(results.items(), key=lambda x: x[1]['ECE']):
        print(f"{method:<20} {metrics['ECE']:>10.4f} {metrics['MCE']:>10.4f}")
    
    return results
