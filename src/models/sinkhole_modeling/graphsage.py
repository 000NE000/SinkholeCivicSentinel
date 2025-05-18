"""
Graph neural network models for sinkhole screening
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, JumpingKnowledge
from torch_geometric.data import Data
import numpy as np
from typing import Tuple, Dict, List, Optional
from src.models.sinkhole_modeling.config import log, GRAPHSAGE_PARAMS


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE implementation for sinkhole risk screening

    Based on "Inductive Representation Learning on Large Graphs" by Hamilton et al.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 jk: str = 'cat'):
        """
        Initialize GraphSAGE model

        Args:
            in_channels: Number of input features
            hidden_channels: Hidden dimension size
            num_layers: Number of graph convolutional layers
            dropout: Dropout probability
            jk: Type of Jumping Knowledge (None, 'cat', 'max', 'lstm')
        """
        super(GraphSAGE, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.jk = jk

        self.convs = nn.ModuleList()
        # First layer from in_channels to hidden_channels
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        # Additional layers from hidden_channels to hidden_channels
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Jumping knowledge connection if specified
        if self.jk:
            if self.jk == 'cat':
                self.jk_layer = JumpingKnowledge(mode='cat')
                final_dim = hidden_channels * num_layers
            elif self.jk == 'max':
                self.jk_layer = JumpingKnowledge(mode='max')
                final_dim = hidden_channels
            elif self.jk == 'lstm':
                self.jk_layer = JumpingKnowledge(mode='lstm', channels=hidden_channels, num_layers=num_layers)
                final_dim = hidden_channels
            else:
                raise ValueError(f"Unknown JK mode: {self.jk}")
        else:
            final_dim = hidden_channels

        # Final classification layer
        self.lin = nn.Linear(final_dim, 1)

    def forward(self, x, edge_index):
        """Forward pass through the network"""
        xs = []  # Store intermediate representations if using JK

        # Apply graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                xs.append(x)

        # Apply jumping knowledge if specified
        if self.jk:
            x = self.jk_layer(xs)

        # Apply final linear layer and sigmoid for binary classification
        x = self.lin(x)
        return torch.sigmoid(x).view(-1)


class UncertaintyMaskedGraphSAGE(torch.nn.Module):
    """
    GraphSAGE with uncertainty masking for sinkhole risk screening

    This model combines GraphSAGE with uncertainty masking to focus
    on uncertain predictions from the stage 1 model.
    """

    def __init__(self,
                 in_channels: int,
                 uncertainty_low: float = 0.4,
                 uncertainty_high: float = 0.6,
                 **kwargs):
        """
        Initialize UncertaintyMaskedGraphSAGE model

        Args:
            in_channels: Number of input features
            uncertainty_low: Lower threshold for uncertainty masking
            uncertainty_high: Upper threshold for uncertainty masking
            **kwargs: Additional arguments for GraphSAGE
        """
        super(UncertaintyMaskedGraphSAGE, self).__init__()

        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high

        # Create base GraphSAGE model
        self.base_model = GraphSAGE(in_channels, **kwargs)

    def forward(self, x, edge_index, stage1_scores=None):
        """
        Forward pass with uncertainty masking

        Args:
            x: Node features
            edge_index: Edge indices
            stage1_scores: Scores from stage 1 model (for uncertainty masking)

        Returns:
            Final scores
        """
        # Get base predictions from GraphSAGE
        base_preds = self.base_model(x, edge_index)

        # If stage1_scores provided, apply uncertainty masking
        if stage1_scores is not None:
            # Convert to tensor if numpy array
            if isinstance(stage1_scores, np.ndarray):
                stage1_scores = torch.tensor(stage1_scores, dtype=torch.float32)

            # Create uncertainty mask
            uncertainty_mask = ((stage1_scores >= self.uncertainty_low) &
                                (stage1_scores <= self.uncertainty_high))

            # Apply mask: use GraphSAGE predictions for uncertain nodes,
            # otherwise use stage1 scores
            final_scores = torch.where(
                uncertainty_mask,
                base_preds,
                stage1_scores
            )

            return final_scores
        else:
            # If no stage1_scores, just return base predictions
            return base_preds


def train_graphsage(data: Data,
                    train_mask: torch.Tensor,
                    val_mask: Optional[torch.Tensor] = None,
                    test_mask: Optional[torch.Tensor] = None,
                    **params) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train GraphSAGE model on PyTorch Geometric Data

    Args:
        data: PyTorch Geometric Data object
        train_mask: Boolean mask for training nodes
        val_mask: Optional boolean mask for validation nodes
        test_mask: Optional boolean mask for test nodes
        **params: Model parameters

    Returns:
        Trained model and training history
    """
    # Set default parameters if not provided
    params = {**GRAPHSAGE_PARAMS, **params}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Initialize model
    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=params['hidden_channels'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        jk=params['jk']
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(params['epochs']):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        # Binary cross entropy loss
        loss = F.binary_cross_entropy(out[train_mask], data.y[train_mask].float())
        loss.backward()
        optimizer.step()

        history['train_loss'].append(loss.item())

        # Validation
        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = F.binary_cross_entropy(val_out[val_mask], data.y[val_mask].float())
                history['val_loss'].append(val_loss.item())

                # Calculate AUC if more than one class in validation set
                if torch.unique(data.y[val_mask]).size(0) > 1:
                    from sklearn.metrics import roc_auc_score
                    try:
                        val_auc = roc_auc_score(data.y[val_mask].cpu().numpy(), val_out[val_mask].cpu().numpy())
                        history['val_auc'].append(val_auc)
                    except:
                        history['val_auc'].append(0.5)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= params['patience']:
                        log(f"Early stopping at epoch {epoch + 1}", level=2)
                        break

        # Log progress
        if (epoch + 1) % 10 == 0:
            log_msg = f"Epoch {epoch + 1}: Train Loss={loss.item():.4f}"
            if val_mask is not None:
                log_msg += f", Val Loss={val_loss.item():.4f}"
                if 'val_auc' in history and history['val_auc']:
                    log_msg += f", Val AUC={history['val_auc'][-1]:.4f}"
            log(log_msg, level=2)

    # Load best model if early stopping occurred
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test evaluation
    if test_mask is not None:
        model.eval()
        with torch.no_grad():
            test_out = model(data.x, data.edge_index)
            test_loss = F.binary_cross_entropy(test_out[test_mask], data.y[test_mask].float())

            # Calculate AUC if more than one class in test set
            if torch.unique(data.y[test_mask]).size(0) > 1:
                from sklearn.metrics import roc_auc_score
                try:
                    test_auc = roc_auc_score(data.y[test_mask].cpu().numpy(), test_out[test_mask].cpu().numpy())
                    log(f"Test: Loss={test_loss.item():.4f}, AUC={test_auc:.4f}", level=1)
                except:
                    log(f"Test: Loss={test_loss.item():.4f}, AUC=N/A", level=1)
            else:
                log(f"Test: Loss={test_loss.item():.4f}", level=1)

    return model, history


def train_uncertainty_masked_graphsage(data: Data,
                                       stage1_scores: torch.Tensor,
                                       train_mask: torch.Tensor,
                                       val_mask: Optional[torch.Tensor] = None,
                                       test_mask: Optional[torch.Tensor] = None,
                                       uncertainty_low: float = 0.4,
                                       uncertainty_high: float = 0.6,
                                       **params) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train UncertaintyMaskedGraphSAGE model on PyTorch Geometric Data

    Args:
        data: PyTorch Geometric Data object
        stage1_scores: Scores from stage 1 model
        train_mask: Boolean mask for training nodes
        val_mask: Optional boolean mask for validation nodes
        test_mask: Optional boolean mask for test nodes
        uncertainty_low: Lower threshold for uncertainty masking
        uncertainty_high: Upper threshold for uncertainty masking
        **params: Model parameters

    Returns:
        Trained model and training history
    """
    # Set default parameters if not provided
    params = {**GRAPHSAGE_PARAMS, **params}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Convert stage1_scores to tensor if numpy array
    if isinstance(stage1_scores, np.ndarray):
        stage1_scores = torch.tensor(stage1_scores, dtype=torch.float32).to(device)

    # Initialize model
    model = UncertaintyMaskedGraphSAGE(
        in_channels=data.num_features,
        uncertainty_low=uncertainty_low,
        uncertainty_high=uncertainty_high,
        hidden_channels=params['hidden_channels'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        jk=params['jk']
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }

    # Create uncertainty mask for training
    uncertainty_mask = ((stage1_scores >= uncertainty_low) &
                        (stage1_scores <= uncertainty_high))

    # Calculate proportion of uncertain samples
    uncertain_ratio = uncertainty_mask.float().mean().item()
    uncertain_train = (uncertainty_mask & train_mask).sum().item()
    log(f"Uncertainty masking: {uncertain_ratio:.2%} of samples are uncertain, {uncertain_train} in training set",
        level=1)

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(params['epochs']):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, stage1_scores)

        # Binary cross entropy loss on uncertain samples
        uncertain_train_mask = uncertainty_mask & train_mask
        if uncertain_train_mask.sum() > 0:
            loss = F.binary_cross_entropy(out[uncertain_train_mask], data.y[uncertain_train_mask].float())
            loss.backward()
            optimizer.step()

            history['train_loss'].append(loss.item())
        else:
            # If no uncertain samples in training set, use base model training
            log("No uncertain samples in training set, using base model training", level=2)
            loss = F.binary_cross_entropy(out[train_mask], data.y[train_mask].float())
            loss.backward()
            optimizer.step()

            history['train_loss'].append(loss.item())

        # Validation
        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index, stage1_scores)

                # Validate on all validation samples
                val_loss = F.binary_cross_entropy(val_out[val_mask], data.y[val_mask].float())
                history['val_loss'].append(val_loss.item())

                # Calculate AUC if more than one class in validation set
                if torch.unique(data.y[val_mask]).size(0) > 1:
                    from sklearn.metrics import roc_auc_score
                    try:
                        val_auc = roc_auc_score(data.y[val_mask].cpu().numpy(), val_out[val_mask].cpu().numpy())
                        history['val_auc'].append(val_auc)
                    except:
                        history['val_auc'].append(0.5)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= params['patience']:
                        log(f"Early stopping at epoch {epoch + 1}", level=2)
                        break

        # Log progress
        if (epoch + 1) % 10 == 0:
            log_msg = f"Epoch {epoch + 1}: Train Loss={loss.item():.4f}"
            if val_mask is not None:
                log_msg += f", Val Loss={val_loss.item():.4f}"
                if 'val_auc' in history and history['val_auc']:
                    log_msg += f", Val AUC={history['val_auc'][-1]:.4f}"
            log(log_msg, level=2)

    # Load best model if early stopping occurred
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test evaluation
    if test_mask is not None:
        model.eval()
        with torch.no_grad():
            test_out = model(data.x, data.edge_index, stage1_scores)
            test_loss = F.binary_cross_entropy(test_out[test_mask], data.y[test_mask].float())

            # Calculate AUC if more than one class in test set
            if torch.unique(data.y[test_mask]).size(0) > 1:
                from sklearn.metrics import roc_auc_score
                try:
                    test_auc = roc_auc_score(data.y[test_mask].cpu().numpy(), test_out[test_mask].cpu().numpy())
                    log(f"Test: Loss={test_loss.item():.4f}, AUC={test_auc:.4f}", level=1)
                except:
                    log(f"Test: Loss={test_loss.item():.4f}, AUC=N/A", level=1)
            else:
                log(f"Test: Loss={test_loss.item():.4f}", level=1)

    return model, history