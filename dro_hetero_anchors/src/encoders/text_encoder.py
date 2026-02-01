"""
Text Encoder for TextCaps - Character/Word-level encoding

Maps text captions (containing OCR text from images) to latent space R^k.
Provides multiple options from simple to sophisticated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNNTextEncoder(nn.Module):
    """Character-level CNN text encoder.
    
    Treats text as a 1D signal and applies convolutions.
    Simple but effective for short texts with OCR content.
    """
    
    def __init__(
        self, 
        latent_dim: int,
        vocab_size: int = 50,  # characters + padding
        embed_dim: int = 32,
        max_len: int = 128,
    ):
        super().__init__()
        
        self.max_len = max_len
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 1D CNN layers
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)  # Fixed output size
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(256 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, max_len) token IDs
        Returns:
            z: (B, latent_dim) latent vectors
        """
        # Embedding: (B, max_len) -> (B, max_len, embed_dim)
        x = self.embedding(x)
        
        # Transpose for Conv1d: (B, embed_dim, max_len)
        x = x.transpose(1, 2)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 128 -> 64
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 64 -> 32
        
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)  # -> 8
        
        # Flatten and project
        x = x.flatten(1)  # (B, 256 * 8)
        z = self.projection(x)  # (B, latent_dim)
        
        return z


class TransformerTextEncoder(nn.Module):
    """Lightweight Transformer-based text encoder.
    
    Uses self-attention to capture relationships between characters/words.
    More powerful but slower than CNN.
    """
    
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int = 50,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_len: int = 128,
    ):
        super().__init__()
        
        self.max_len = max_len
        self.embed_dim = embed_dim
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, max_len) token IDs
        Returns:
            z: (B, latent_dim) latent vectors
        """
        B, L = x.shape
        
        # Create position indices
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        
        # Embeddings
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        # Create padding mask (True where padded)
        # Assuming 0 is padding token
        padding_mask = (x.sum(dim=-1) == 0)  # (B, L)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Pool over sequence (mean of non-padded tokens)
        mask = (~padding_mask).unsqueeze(-1).float()  # (B, L, 1)
        x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B, embed_dim)
        
        # Project to latent space
        z = self.projection(x)
        
        return z


class MLPTextEncoder(nn.Module):
    """Simple MLP-based text encoder (bag-of-characters).
    
    Fastest option - just sums character embeddings and applies MLP.
    Good for quick iteration.
    """
    
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int = 50,
        embed_dim: int = 64,
        max_len: int = 128,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, max_len) token IDs
        Returns:
            z: (B, latent_dim) latent vectors
        """
        # Embedding: (B, max_len, embed_dim)
        x = self.embedding(x)
        
        # Mean pooling over sequence (ignoring padding)
        mask = (x.sum(dim=-1) != 0).unsqueeze(-1).float()  # (B, max_len, 1)
        x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B, embed_dim)
        
        # MLP
        z = self.mlp(x)
        
        return z
