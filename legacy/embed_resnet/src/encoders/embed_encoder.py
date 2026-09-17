"""Multi-view mammography encoder for EMBED.

Maps an exam-breast's available views → a single shared-latent vector.

Two modes (this is the "common vs per-group encoder" ablation axis, mirroring the
tabular datasets' `common_encoder` flag):

  * SHARED backbone (per_view_encoders=False, default): one ResNet encodes every
    view; a learnable per-view-type embedding tags which of the 4 canonical views
    (FFDM/C-View × CC/MLO) each came from. Parameters are shared across modalities.

  * PER-VIEW backbones (per_view_encoders=True): each of the 4 view-types gets its
    OWN ResNet — i.e. a genuine per-modality encoder φ_g. Views are then pooled into
    the shared latent. ~4× the encoder params/compute; the true "per-group encoder"
    arm of our method.

In both modes, present-view latents are masked-mean-pooled and passed through a
SHARED projection into the common latent space (so class anchors + GroupDRO operate
on one aligned latent regardless of which views an exam has). Missing modalities are
handled natively — an exam with 2 views simply pools over those 2.

Input:  views (B, V, 3, H, W), mask (B, V)   with V = NUM_VIEW_TYPES (4)
Output: z (B, latent_dim)
"""

import torch
import torch.nn as nn
from torchvision import models


def _make_backbone(backbone: str, pretrained: bool):
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    feat_dim = net.fc.in_features            # 512 (r18) / 2048 (r50)
    net.fc = nn.Identity()
    return net, feat_dim


class MammoMultiViewEncoder(nn.Module):
    def __init__(self, latent_dim: int, num_view_types: int = 4,
                 pretrained: bool = True, dropout: float = 0.2,
                 backbone: str = "resnet18", per_view_encoders: bool = False):
        super().__init__()
        self.num_view_types = num_view_types
        self.per_view_encoders = per_view_encoders

        if per_view_encoders:
            # One dedicated backbone per view-type (per-group encoder φ_g)
            mades = [_make_backbone(backbone, pretrained) for _ in range(num_view_types)]
            self.backbones = nn.ModuleList([m[0] for m in mades])
            feat_dim = mades[0][1]
            self.view_embed = None
        else:
            net, feat_dim = _make_backbone(backbone, pretrained)
            self.backbone = net
            # Learnable per-view-type tag added to each view's shared-backbone features
            self.view_embed = nn.Parameter(torch.zeros(num_view_types, feat_dim))
            nn.init.normal_(self.view_embed, std=0.02)

        # SHARED projection into the common latent (both modes)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(256, latent_dim),
        )

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, V, C, H, W = views.shape
        if self.per_view_encoders:
            # Encode each view-type with its own backbone
            feats = torch.stack([
                self.backbones[v](views[:, v]) for v in range(V)
            ], dim=1)                                                   # (B,V,F)
        else:
            feats = self.backbone(views.reshape(B * V, C, H, W)).reshape(B, V, -1)
            feats = feats + self.view_embed.unsqueeze(0)               # tag view type
        m = mask.unsqueeze(-1)                                         # (B,V,1)
        pooled = (feats * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # masked mean
        return self.projection(pooled)                                # (B, latent)
