import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler
from collections import Counter


class ClassBalancedSampler(Sampler):
    """
    Class-balanced sampler for multi-label classification.
    Samples based on inverse class frequency.
    """
    
    def __init__(self, labels, class_counts, num_samples=None):
        """
        Args:
            labels: numpy array of shape (N, num_classes) with binary labels
            class_counts: array of class counts
            num_samples: number of samples per epoch (default: len(labels))
        """
        self.labels = labels
        self.num_samples = num_samples if num_samples else len(labels)
        
        # Calculate sample weights based on rarest class in each sample
        class_weights = 1.0 / (np.array(class_counts) + 1e-6)
        class_weights = class_weights / class_weights.sum()
        
        sample_weights = []
        for i in range(len(labels)):
            # Weight by the rarest positive class in this sample
            positive_classes = np.where(labels[i] > 0)[0]
            if len(positive_classes) > 0:
                weight = max(class_weights[positive_classes])
            else:
                weight = class_weights.mean()
            sample_weights.append(weight)
        
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
        
    def __iter__(self):
        indices = torch.multinomial(
            self.sample_weights, 
            self.num_samples, 
            replacement=True
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples


class ProgressiveBalancedSampler(Sampler):
    """
    Progressive sampling that gradually shifts from natural to balanced distribution.
    Useful for curriculum learning approach.
    """
    
    def __init__(self, labels, class_counts, num_samples=None, max_epochs=30):
        self.labels = labels
        self.num_samples = num_samples if num_samples else len(labels)
        self.max_epochs = max_epochs
        self.current_epoch = 0
        
        # Natural weights (uniform)
        self.natural_weights = torch.ones(len(labels), dtype=torch.float64)
        self.natural_weights = self.natural_weights / self.natural_weights.sum()
        
        # Balanced weights
        class_weights = 1.0 / (np.array(class_counts) + 1e-6)
        class_weights = class_weights / class_weights.sum()
        
        balanced_weights = []
        for i in range(len(labels)):
            positive_classes = np.where(labels[i] > 0)[0]
            if len(positive_classes) > 0:
                weight = max(class_weights[positive_classes])
            else:
                weight = class_weights.mean()
            balanced_weights.append(weight)
        
        self.balanced_weights = torch.tensor(balanced_weights, dtype=torch.float64)
        self.balanced_weights = self.balanced_weights / self.balanced_weights.sum()
        
    def update_epoch(self, epoch):
        self.current_epoch = epoch
        
    def __iter__(self):
        # Linearly interpolate between natural and balanced
        alpha = min(self.current_epoch / self.max_epochs, 1.0)
        weights = (1 - alpha) * self.natural_weights + alpha * self.balanced_weights
        
        indices = torch.multinomial(weights, self.num_samples, replacement=True)
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples


class SquareRootSampler(Sampler):
    """
    Square-root sampling - a middle ground between natural and balanced.
    Often works better than pure class-balanced sampling.
    """
    
    def __init__(self, labels, class_counts, num_samples=None):
        self.labels = labels
        self.num_samples = num_samples if num_samples else len(labels)
        
        # Square root of inverse frequency
        class_weights = 1.0 / np.sqrt(np.array(class_counts) + 1e-6)
        class_weights = class_weights / class_weights.sum()
        
        sample_weights = []
        for i in range(len(labels)):
            positive_classes = np.where(labels[i] > 0)[0]
            if len(positive_classes) > 0:
                weight = max(class_weights[positive_classes])
            else:
                weight = class_weights.mean()
            sample_weights.append(weight)
        
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
        
    def __iter__(self):
        indices = torch.multinomial(
            self.sample_weights, 
            self.num_samples, 
            replacement=True
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples


def get_sampler(sampler_type, labels, class_counts, num_samples=None):
    """Factory function to get sampler."""
    if sampler_type == "default" or sampler_type is None:
        return None  # Use default shuffle
    elif sampler_type == "class_balanced":
        return ClassBalancedSampler(labels, class_counts, num_samples)
    elif sampler_type == "sqrt":
        return SquareRootSampler(labels, class_counts, num_samples)
    elif sampler_type == "progressive":
        return ProgressiveBalancedSampler(labels, class_counts, num_samples)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
