"""
ML-GCN: Multi-Label Image Recognition with Graph Convolutional Networks

Paper: https://arxiv.org/abs/1904.03582

Key Insight: Labels in multi-label classification are not independent.
For CXR, diseases often co-occur (e.g., Cardiomegaly + Pleural Effusion).
ML-GCN uses a GCN to model label dependencies and improve predictions.

Architecture:
1. CNN backbone extracts image features
2. GCN learns label embeddings using co-occurrence graph
3. Classifier combines image features with label embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Optional, Tuple

from .base import BaseClassifier


class GraphConvolution(nn.Module):
    """
    Simple Graph Convolution layer.
    H' = sigma(A * H * W)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
        Returns:
            Updated node features [num_nodes, out_features]
        """
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class MLGCN(BaseClassifier):
    """
    Multi-Label Graph Convolutional Network.
    
    Uses GCN to model label co-occurrence relationships.
    The adjacency matrix is built from training data label statistics.
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        adj_matrix: Optional[np.ndarray] = None,
        word_embeddings: Optional[np.ndarray] = None,
        embed_dim: int = 300,
        hidden_dim: int = 1024,
        dropout: float = 0.5,
        t: float = 0.4,  # Threshold for adjacency matrix
    ):
        super().__init__(num_classes, pretrained)
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.t = t
        
        # Initialize backbone
        if backbone == "resnet50":
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V2
                self.backbone = models.resnet50(weights=weights)
            else:
                self.backbone = models.resnet50(weights=None)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == "resnet101":
            if pretrained:
                weights = models.ResNet101_Weights.IMAGENET1K_V2
                self.backbone = models.resnet101(weights=weights)
            else:
                self.backbone = models.resnet101(weights=None)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == "densenet121":
            if pretrained:
                weights = models.DenseNet121_Weights.IMAGENET1K_V1
                self.backbone = models.densenet121(weights=weights)
            else:
                self.backbone = models.densenet121(weights=None)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Word embeddings for labels (initialized randomly if not provided)
        if word_embeddings is not None:
            self.word_embeddings = nn.Parameter(
                torch.FloatTensor(word_embeddings),
                requires_grad=False
            )
            embed_dim = word_embeddings.shape[1]
        else:
            self.word_embeddings = nn.Parameter(
                torch.FloatTensor(num_classes, embed_dim),
                requires_grad=True
            )
            nn.init.xavier_uniform_(self.word_embeddings)
        
        # GCN layers
        self.gc1 = GraphConvolution(embed_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, self.feature_dim)
        
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # Adjacency matrix (will be set from training data)
        if adj_matrix is not None:
            self.register_buffer('adj', torch.FloatTensor(self._normalize_adj(adj_matrix)))
        else:
            # Identity matrix as placeholder
            self.register_buffer('adj', torch.eye(num_classes))
    
    def _normalize_adj(self, adj: np.ndarray) -> np.ndarray:
        """Normalize adjacency matrix."""
        # Apply threshold
        adj = np.where(adj > self.t, adj, 0)
        
        # Add self-loops
        adj = adj + np.eye(adj.shape[0])
        
        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        d = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        return adj_normalized
    
    def set_adjacency_matrix(self, adj_matrix: np.ndarray):
        """Set adjacency matrix from training data."""
        adj_normalized = self._normalize_adj(adj_matrix)
        self.adj = torch.FloatTensor(adj_normalized).to(self.word_embeddings.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, 3, H, W]
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # Extract image features
        img_features = self.backbone(x)  # [batch_size, feature_dim]
        
        # GCN forward pass
        # Input: word embeddings [num_classes, embed_dim]
        # Output: label classifiers [num_classes, feature_dim]
        
        gcn_out = self.gc1(self.word_embeddings, self.adj)
        gcn_out = self.relu(gcn_out)
        gcn_out = self.dropout(gcn_out)
        gcn_out = self.gc2(gcn_out, self.adj)  # [num_classes, feature_dim]
        
        # Transpose for matrix multiplication
        gcn_out = gcn_out.transpose(0, 1)  # [feature_dim, num_classes]
        
        # Compute logits: image_features @ label_classifiers
        logits = torch.matmul(img_features, gcn_out)  # [batch_size, num_classes]
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_adjacency_matrix(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Build adjacency matrix from training labels based on co-occurrence.
    
    Args:
        labels: Training labels [num_samples, num_classes] binary matrix
        num_classes: Number of classes
    
    Returns:
        adj: Adjacency matrix [num_classes, num_classes]
             adj[i,j] = P(label_j | label_i) = count(i AND j) / count(i)
    """
    # Count co-occurrences
    cooccurrence = labels.T @ labels  # [num_classes, num_classes]
    
    # Count individual occurrences
    counts = labels.sum(axis=0) + 1e-6  # [num_classes,] avoid division by zero
    
    # Compute conditional probability: P(j|i) = count(i,j) / count(i)
    adj = cooccurrence / counts.reshape(-1, 1)
    
    return adj


def build_adjacency_matrix_symmetric(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Build symmetric adjacency matrix using Jaccard similarity.
    
    adj[i,j] = |i AND j| / |i OR j|
    """
    # Co-occurrence count
    cooccurrence = labels.T @ labels  # [num_classes, num_classes]
    
    # Individual counts
    counts = labels.sum(axis=0)  # [num_classes,]
    
    # Union count: |i| + |j| - |i AND j|
    union = counts.reshape(-1, 1) + counts.reshape(1, -1) - cooccurrence
    union = np.maximum(union, 1e-6)  # Avoid division by zero
    
    # Jaccard similarity
    adj = cooccurrence / union
    
    return adj


class MLGCNWithAttention(BaseClassifier):
    """
    ML-GCN with attention mechanism for better feature-label interaction.
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        adj_matrix: Optional[np.ndarray] = None,
        embed_dim: int = 300,
        hidden_dim: int = 1024,
        dropout: float = 0.5,
        t: float = 0.4,
    ):
        super().__init__(num_classes, pretrained)
        
        self.num_classes = num_classes
        self.t = t
        
        # Backbone
        if backbone == "resnet50":
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V2
                base = models.resnet50(weights=weights)
            else:
                base = models.resnet50(weights=None)
            
            # Remove final pooling and FC
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.feature_dim = 2048
        else:
            raise ValueError(f"Backbone {backbone} not yet supported for attention variant")
        
        # Spatial dimensions after backbone (for 512x512 input)
        self.spatial_size = 16  # 512 / 32
        
        # Word embeddings
        self.word_embeddings = nn.Parameter(
            torch.FloatTensor(num_classes, embed_dim),
            requires_grad=True
        )
        nn.init.xavier_uniform_(self.word_embeddings)
        
        # GCN layers
        self.gc1 = GraphConvolution(embed_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, self.feature_dim)
        
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # Attention projection
        self.attention_proj = nn.Linear(self.feature_dim, self.feature_dim)
        
        # Adjacency matrix
        if adj_matrix is not None:
            self.register_buffer('adj', torch.FloatTensor(self._normalize_adj(adj_matrix)))
        else:
            self.register_buffer('adj', torch.eye(num_classes))
    
    def _normalize_adj(self, adj: np.ndarray) -> np.ndarray:
        adj = np.where(adj > self.t, adj, 0)
        adj = adj + np.eye(adj.shape[0])
        d = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    
    def set_adjacency_matrix(self, adj_matrix: np.ndarray):
        adj_normalized = self._normalize_adj(adj_matrix)
        self.adj = torch.FloatTensor(adj_normalized).to(self.word_embeddings.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Extract spatial features [batch, feature_dim, H, W]
        spatial_features = self.backbone(x)
        
        # Global average pooling for image features
        img_features = F.adaptive_avg_pool2d(spatial_features, 1)
        img_features = img_features.view(batch_size, -1)  # [batch, feature_dim]
        
        # GCN forward
        gcn_out = self.gc1(self.word_embeddings, self.adj)
        gcn_out = self.relu(gcn_out)
        gcn_out = self.dropout(gcn_out)
        gcn_out = self.gc2(gcn_out, self.adj)  # [num_classes, feature_dim]
        
        # Attention: use label embeddings to attend to spatial features
        # Reshape spatial features: [batch, feature_dim, H*W] -> [batch, H*W, feature_dim]
        B, C, H, W = spatial_features.shape
        spatial_flat = spatial_features.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
        
        # Compute attention weights: [batch, num_classes, H*W]
        gcn_proj = self.attention_proj(gcn_out)  # [num_classes, feature_dim]
        attention = torch.matmul(gcn_proj, spatial_flat.permute(0, 2, 1))  # [batch, num_classes, H*W]
        attention = F.softmax(attention, dim=-1)
        
        # Weighted sum of spatial features: [batch, num_classes, feature_dim]
        attended_features = torch.matmul(attention, spatial_flat)
        
        # Compute logits via dot product with GCN outputs
        logits = (attended_features * gcn_out.unsqueeze(0)).sum(dim=-1)  # [batch, num_classes]
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return F.adaptive_avg_pool2d(features, 1).view(x.size(0), -1)
