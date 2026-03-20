"""MLP encoder for tabular data (Fed-Heart Disease, etc.)."""

import torch
import torch.nn as nn


class MLPTabularEncoder(nn.Module):
    """MLP encoder for tabular features.
    
    Architecture: input_dim → hidden → hidden → latent_dim
    Each hidden layer has BatchNorm and ReLU.
    
    Args:
        latent_dim: Output embedding dimension
        input_dim: Number of input features (default 13 for Fed-Heart Disease)
        hidden_dim: Hidden layer dimension (default 64)
        dropout: Dropout probability (default 0.1)
    """
    
    def __init__(self, latent_dim: int, input_dim: int = 13, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, latent_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Latent embedding of shape (batch_size, latent_dim)
        """
        return self.encoder(x)


class MLPTabularEncoderLarge(nn.Module):
    """Larger MLP encoder for tabular features with more capacity.
    
    Architecture: input_dim → 128 → 64 → latent_dim
    
    Args:
        latent_dim: Output embedding dimension
        input_dim: Number of input features (default 13 for Fed-Heart Disease)
        dropout: Dropout probability (default 0.1)
    """
    
    def __init__(self, latent_dim: int, input_dim: int = 13, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(64, latent_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MLPTabularEncoderLN(nn.Module):
    """MLP encoder with LayerNorm (works with batch size 1).
    
    Use this for extreme imbalance experiments where some groups have very few samples.
    Architecture: input_dim → hidden → hidden → latent_dim
    
    Args:
        latent_dim: Output embedding dimension
        input_dim: Number of input features (default 13 for Fed-Heart Disease)
        hidden_dim: Hidden layer dimension (default 64)
        dropout: Dropout probability (default 0.1)
    """
    
    def __init__(self, latent_dim: int, input_dim: int = 13, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, latent_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MLPTabularEncoderDeep(nn.Module):
    """Deeper MLP encoder with LayerNorm (4 hidden layers).
    
    Larger capacity model that can overfit to majority groups,
    making GroupDRO's reweighting more impactful.
    Architecture: input_dim → 128 → 128 → 64 → 64 → latent_dim
    
    Args:
        latent_dim: Output embedding dimension
        input_dim: Number of input features (default 13 for Fed-Heart Disease)
        dropout: Dropout probability (default 0.1)
    """
    
    def __init__(self, latent_dim: int, input_dim: int = 13, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
