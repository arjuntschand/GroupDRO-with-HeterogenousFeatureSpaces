"""
Visual Encoder for TextCaps - ResNet-based

Maps RGB images (224x224x3) to latent space R^k.
Uses a lightweight ResNet18 backbone with custom head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetVisualEncoder(nn.Module):
    """ResNet18-based visual encoder for TextCaps images.
    
    Takes 224x224 RGB images and outputs latent_dim features.
    Uses pretrained weights by default for better transfer learning.
    """
    
    def __init__(self, latent_dim: int, pretrained: bool = True):
        super().__init__()
        
        # Load ResNet18 backbone
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Get the feature dimension before the final FC layer
        backbone_dim = self.backbone.fc.in_features  # 512 for ResNet18
        
        # Replace final FC with identity (we'll add our own)
        self.backbone.fc = nn.Identity()
        
        # Custom projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) RGB images
        Returns:
            z: (B, latent_dim) latent vectors
        """
        features = self.backbone(x)  # (B, 512)
        z = self.projection(features)  # (B, latent_dim)
        return z


class SimpleCNNVisualEncoder(nn.Module):
    """Lighter CNN encoder for faster iteration.
    
    Good for initial experiments when you don't need full ResNet.
    """
    
    def __init__(self, latent_dim: int):
        super().__init__()
        
        # Simple CNN backbone
        self.features = nn.Sequential(
            # Conv block 1: 224 -> 112
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 2: 112 -> 56
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 3: 56 -> 28
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 4: 28 -> 14
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 5: 14 -> 7
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        z = self.projection(features)
        return z
