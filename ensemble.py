"""
Ensemble Utilities for CXR-LT Competition

Provides various ensemble strategies for combining predictions from multiple models:
1. Simple averaging
2. Weighted averaging
3. Rank-based averaging
4. Stacking (meta-learner)
5. Test-Time Augmentation (TTA)

For competitions, ensembling diverse models typically improves mAP by 1-3%.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def average_predictions(predictions_list: List[np.ndarray]) -> np.ndarray:
    """
    Simple averaging of predictions from multiple models.
    
    Args:
        predictions_list: List of prediction arrays, each [num_samples, num_classes]
    
    Returns:
        Averaged predictions [num_samples, num_classes]
    """
    return np.mean(predictions_list, axis=0)


def weighted_average_predictions(
    predictions_list: List[np.ndarray],
    weights: List[float]
) -> np.ndarray:
    """
    Weighted averaging of predictions.
    
    Args:
        predictions_list: List of prediction arrays
        weights: Weight for each model (should sum to 1, or will be normalized)
    
    Returns:
        Weighted averaged predictions
    """
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    
    weighted_preds = np.zeros_like(predictions_list[0])
    for pred, w in zip(predictions_list, weights):
        weighted_preds += w * pred
    
    return weighted_preds


def geometric_mean_predictions(predictions_list: List[np.ndarray]) -> np.ndarray:
    """
    Geometric mean of predictions (useful for probabilities).
    """
    log_preds = [np.log(p + 1e-8) for p in predictions_list]
    mean_log = np.mean(log_preds, axis=0)
    return np.exp(mean_log)


def rank_average_predictions(predictions_list: List[np.ndarray]) -> np.ndarray:
    """
    Rank-based averaging (robust to different score scales).
    
    Each model's predictions are converted to ranks, then averaged.
    """
    num_samples, num_classes = predictions_list[0].shape
    ranked_preds = []
    
    for preds in predictions_list:
        # Convert to ranks per class
        ranks = np.zeros_like(preds)
        for c in range(num_classes):
            ranks[:, c] = preds[:, c].argsort().argsort()
        # Normalize to [0, 1]
        ranks = ranks / (num_samples - 1)
        ranked_preds.append(ranks)
    
    return np.mean(ranked_preds, axis=0)


def power_average_predictions(
    predictions_list: List[np.ndarray],
    power: float = 2.0
) -> np.ndarray:
    """
    Power averaging - emphasizes higher predictions.
    Useful when some models are more confident.
    """
    powered = [np.power(p, power) for p in predictions_list]
    mean_powered = np.mean(powered, axis=0)
    return np.power(mean_powered, 1.0 / power)


class ModelEnsemble(nn.Module):
    """
    Ensemble wrapper that runs multiple models and combines predictions.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        ensemble_method: str = "average"
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
        self.ensemble_method = ensemble_method
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(x))
                predictions.append(pred)
        
        # Stack predictions: [num_models, batch_size, num_classes]
        stacked = torch.stack(predictions, dim=0)
        
        if self.ensemble_method == "average":
            return stacked.mean(dim=0)
        elif self.ensemble_method == "weighted":
            weights = torch.tensor(self.weights, device=x.device).view(-1, 1, 1)
            return (stacked * weights).sum(dim=0)
        elif self.ensemble_method == "max":
            return stacked.max(dim=0)[0]
        else:
            return stacked.mean(dim=0)


class TTAWrapper(nn.Module):
    """
    Test-Time Augmentation wrapper.
    
    Applies multiple augmentations at test time and averages predictions.
    Common augmentations for CXR:
    - Horizontal flip
    - Minor rotation
    - Scale variations
    """
    
    def __init__(
        self,
        model: nn.Module,
        tta_transforms: Optional[List[Callable]] = None,
        merge_mode: str = "mean"
    ):
        super().__init__()
        self.model = model
        self.merge_mode = merge_mode
        
        if tta_transforms is None:
            # Default TTA: original + horizontal flip
            self.tta_transforms = [
                lambda x: x,  # Original
                lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
            ]
        else:
            self.tta_transforms = tta_transforms
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for transform in self.tta_transforms:
                augmented = transform(x)
                pred = torch.sigmoid(self.model(augmented))
                predictions.append(pred)
        
        stacked = torch.stack(predictions, dim=0)
        
        if self.merge_mode == "mean":
            return stacked.mean(dim=0)
        elif self.merge_mode == "max":
            return stacked.max(dim=0)[0]
        elif self.merge_mode == "gmean":
            # Geometric mean
            log_preds = torch.log(stacked + 1e-8)
            return torch.exp(log_preds.mean(dim=0))
        else:
            return stacked.mean(dim=0)


def ensemble_predict(
    models: List[nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    weights: Optional[List[float]] = None,
    use_tta: bool = False
) -> np.ndarray:
    """
    Run ensemble prediction on a dataloader.
    
    Args:
        models: List of trained models
        dataloader: Test dataloader
        device: Device to run on
        weights: Optional weights for each model
        use_tta: Whether to use test-time augmentation
    
    Returns:
        Ensemble predictions [num_samples, num_classes]
    """
    all_predictions = [[] for _ in range(len(models))]
    
    # Wrap models with TTA if requested
    if use_tta:
        models = [TTAWrapper(m) for m in models]
    
    for model in models:
        model.eval()
        model.to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Ensemble prediction"):
            if isinstance(batch, dict):
                images = batch['image'].to(device)
            else:
                images = batch[0].to(device)
            
            for i, model in enumerate(models):
                pred = torch.sigmoid(model(images))
                all_predictions[i].append(pred.cpu().numpy())
    
    # Concatenate predictions for each model
    predictions_list = [np.concatenate(preds, axis=0) for preds in all_predictions]
    
    # Ensemble
    if weights is not None:
        return weighted_average_predictions(predictions_list, weights)
    else:
        return average_predictions(predictions_list)


def load_models_from_checkpoints(
    checkpoint_paths: List[str],
    model_classes: List[type],
    num_classes: int,
    device: torch.device
) -> List[nn.Module]:
    """
    Load multiple models from checkpoint files.
    
    Args:
        checkpoint_paths: Paths to checkpoint files
        model_classes: Model class for each checkpoint
        num_classes: Number of output classes
        device: Device to load models to
    
    Returns:
        List of loaded models
    """
    models = []
    
    for path, model_class in zip(checkpoint_paths, model_classes):
        model = model_class(num_classes=num_classes, pretrained=False)
        checkpoint = torch.load(path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        models.append(model)
        print(f"Loaded model from {path}")
    
    return models


def optimize_ensemble_weights(
    predictions_list: List[np.ndarray],
    labels: np.ndarray,
    metric_fn: Callable,
    num_iterations: int = 1000
) -> Tuple[List[float], float]:
    """
    Find optimal ensemble weights using random search.
    
    Args:
        predictions_list: Predictions from each model
        labels: Ground truth labels
        metric_fn: Function that computes metric (higher is better)
        num_iterations: Number of random weight combinations to try
    
    Returns:
        Tuple of (optimal_weights, best_score)
    """
    num_models = len(predictions_list)
    best_weights = [1.0 / num_models] * num_models
    best_score = metric_fn(average_predictions(predictions_list), labels)
    
    print(f"Initial score (equal weights): {best_score:.4f}")
    
    for i in range(num_iterations):
        # Generate random weights
        weights = np.random.dirichlet(np.ones(num_models))
        
        # Compute ensemble prediction
        ensemble_pred = weighted_average_predictions(predictions_list, weights.tolist())
        
        # Compute score
        score = metric_fn(ensemble_pred, labels)
        
        if score > best_score:
            best_score = score
            best_weights = weights.tolist()
            print(f"Iteration {i}: New best score {best_score:.4f}, weights: {best_weights}")
    
    return best_weights, best_score


def blend_submission_files(
    submission_paths: List[str],
    weights: Optional[List[float]] = None,
    output_path: str = "blended_submission.csv"
) -> pd.DataFrame:
    """
    Blend multiple submission CSV files.
    
    Args:
        submission_paths: Paths to submission CSV files
        weights: Optional weights for each submission
        output_path: Path for output blended submission
    
    Returns:
        Blended submission DataFrame
    """
    # Load all submissions
    submissions = [pd.read_csv(path) for path in submission_paths]
    
    # Get ID column (first column)
    id_col = submissions[0].columns[0]
    
    # Get prediction columns (all except ID)
    pred_cols = [c for c in submissions[0].columns if c != id_col]
    
    # Verify all submissions have same structure
    for i, sub in enumerate(submissions):
        if list(sub.columns) != list(submissions[0].columns):
            raise ValueError(f"Submission {i} has different columns")
        if len(sub) != len(submissions[0]):
            raise ValueError(f"Submission {i} has different length")
    
    # Extract predictions
    predictions_list = [sub[pred_cols].values for sub in submissions]
    
    # Blend
    if weights is not None:
        blended = weighted_average_predictions(predictions_list, weights)
    else:
        blended = average_predictions(predictions_list)
    
    # Create output DataFrame
    result = pd.DataFrame({id_col: submissions[0][id_col]})
    for i, col in enumerate(pred_cols):
        result[col] = blended[:, i]
    
    result.to_csv(output_path, index=False)
    print(f"Blended submission saved to {output_path}")
    
    return result


class StackingEnsemble:
    """
    Stacking ensemble - trains a meta-learner on base model predictions.
    """
    
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_learner: Optional[nn.Module] = None,
        num_classes: int = 30
    ):
        self.base_models = base_models
        self.num_classes = num_classes
        
        if meta_learner is None:
            # Simple linear meta-learner
            self.meta_learner = nn.Sequential(
                nn.Linear(num_classes * len(base_models), 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            self.meta_learner = meta_learner
    
    def get_base_predictions(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> np.ndarray:
        """Get predictions from all base models."""
        all_preds = []
        
        for model in self.base_models:
            model.eval()
            model.to(device)
            
            preds = []
            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, dict):
                        images = batch['image'].to(device)
                    else:
                        images = batch[0].to(device)
                    
                    pred = torch.sigmoid(model(images))
                    preds.append(pred.cpu().numpy())
            
            all_preds.append(np.concatenate(preds, axis=0))
        
        # Concatenate along feature dimension
        return np.concatenate(all_preds, axis=1)
    
    def fit_meta_learner(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epochs: int = 10,
        lr: float = 1e-3
    ):
        """Train the meta-learner on base model predictions."""
        # Get base predictions
        train_features = self.get_base_predictions(train_loader, device)
        
        # Get labels
        train_labels = []
        for batch in train_loader:
            if isinstance(batch, dict):
                train_labels.append(batch['labels'].numpy())
            else:
                train_labels.append(batch[1].numpy())
        train_labels = np.concatenate(train_labels, axis=0)
        
        # Train meta-learner
        self.meta_learner.to(device)
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        train_features = torch.FloatTensor(train_features).to(device)
        train_labels = torch.FloatTensor(train_labels).to(device)
        
        dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(epochs):
            self.meta_learner.train()
            total_loss = 0
            
            for features, labels in loader:
                optimizer.zero_grad()
                outputs = self.meta_learner(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> np.ndarray:
        """Get stacked predictions."""
        base_preds = self.get_base_predictions(dataloader, device)
        base_preds = torch.FloatTensor(base_preds).to(device)
        
        self.meta_learner.eval()
        with torch.no_grad():
            meta_preds = torch.sigmoid(self.meta_learner(base_preds))
        
        return meta_preds.cpu().numpy()
