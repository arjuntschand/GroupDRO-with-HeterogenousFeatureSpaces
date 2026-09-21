"""Xenia's authoritative EMBED architecture (per EMBED_Experiments_Description.docx).

Pipeline (all downstream of a FROZEN ViT-Base that produces a cached 768-d CLS
embedding per image):

  per-view Linear(768 -> 64)  [shared across groups]
    -> per-group MLP_g: Linear(|g|*64 -> 64), GELU, Linear(64 -> 64)   [z in R^64]
    -> shared head Linear(64 -> 4)
    -> 4 diagonal-Gaussian class anchors  N(m_c, diag(s_c)),  s_c = softplus(rho_c)+1e-4

The four view types (modalities):
  M1 = C-View CC, M2 = C-View MLO, M3 = FFDM CC, M4 = FFDM MLO

The six groups (by which views are present), with |g| = #views:
  g1={M3}(1)  g2={M1,M3}(2)  g3={M4}(1)  g4={M3,M4}(2)  g5={M1,M3,M4}(3)  g6={M1,M2,M3,M4}(4)

Losses (all closed-form / cheap on cached vectors):
  - task  : cross-entropy on head logits
  - fit   : W2^2 between per-class batch moments and diagonal anchor moments (Eq. below)
  - sep   : reparameterized samples from each anchor pushed through the head, CE vs class
"""
from __future__ import annotations
from typing import Dict, List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

VIEWS = ["M1", "M2", "M3", "M4"]                     # canonical view order
GROUP_VIEWS: Dict[str, List[str]] = {               # views present in each group
    "g1": ["M3"],
    "g2": ["M1", "M3"],
    "g3": ["M4"],
    "g4": ["M3", "M4"],
    "g5": ["M1", "M3", "M4"],
    "g6": ["M1", "M2", "M3", "M4"],
}
GROUPS = ["g1", "g2", "g3", "g4", "g5", "g6"]

# No-overlap variant (2026-09-21): four groups, one view each, no view shared by two groups, so
# nothing is shared at the input (the per-view projections below are then per-group too). Fixed
# before any run: every group keeps the lowest-numbered view that no earlier group has taken;
# g4 and g5 are dropped because four views cannot give six disjoint groups.
DISJOINT_GROUP_VIEWS: Dict[str, List[str]] = {"g1": ["M3"], "g2": ["M1"], "g3": ["M4"], "g6": ["M2"]}


def use_disjoint_views() -> None:
    """Switch the module-level group definition in place, so every importer sees it. Call before
    loading data or building a model."""
    GROUP_VIEWS.clear(); GROUP_VIEWS.update(DISJOINT_GROUP_VIEWS)
    GROUPS[:] = list(DISJOINT_GROUP_VIEWS)


class DiagAnchors(nn.Module):
    """Per-class diagonal Gaussian anchors N(m_c, diag(s_c)), s_c = softplus(rho_c)+eps."""

    def __init__(self, num_classes: int, dim: int, eps: float = 1e-4):
        super().__init__()
        self.num_classes, self.dim, self.eps = num_classes, dim, eps
        # Xenia spec: m_c ~ Normal(std 0.1); rho_c s.t. s_c = softplus(rho)+eps starts ~1
        # softplus(rho)=1 -> rho = log(e^1 - 1) ~= 0.5413
        self.m = nn.Parameter(torch.randn(num_classes, dim) * 0.1)
        self.rho = nn.Parameter(torch.full((num_classes, dim), 0.5413))

    def var(self) -> torch.Tensor:                  # s_c  (C, D), strictly positive
        return F.softplus(self.rho) + self.eps


class XeniaEmbedModel(nn.Module):
    """Per-view projections + per-group MLPs + shared head + diagonal anchors."""

    def __init__(self, in_dim: int = 768, proj_dim: int = 64, latent_dim: int = 64,
                 num_classes: int = 4, anchor_eps: float = 1e-4):
        super().__init__()
        self.proj_dim, self.latent_dim, self.num_classes = proj_dim, latent_dim, num_classes
        # shared per-view projections 768 -> 64
        self.proj = nn.ModuleDict({v: nn.Linear(in_dim, proj_dim) for v in VIEWS})
        # per-group MLPs; input dim = |g| * proj_dim
        self.mlp = nn.ModuleDict()
        for g, vs in GROUP_VIEWS.items():
            self.mlp[g] = nn.Sequential(
                nn.Linear(len(vs) * proj_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
        self.head = nn.Linear(latent_dim, num_classes)          # shared classifier
        self.anchors = DiagAnchors(num_classes, latent_dim, eps=anchor_eps)

    def encode(self, group: str, view_feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """view_feats: {view -> (B, 768)} for exactly the views present in `group`.
        Returns latent z (B, latent_dim)."""
        parts = [self.proj[v](view_feats[v]) for v in GROUP_VIEWS[group]]  # canonical order
        x = torch.cat(parts, dim=1)
        return self.mlp[group](x)

    def forward(self, group: str, view_feats: Dict[str, torch.Tensor]):
        z = self.encode(group, view_feats)
        return self.head(z), z


# ---------------------------------------------------------------- losses ----

def anchor_fit_loss(z: torch.Tensor, y: torch.Tensor, anchors: DiagAnchors,
                    min_count: int = 8) -> torch.Tensor:
    """W2^2 between per-class batch moments and diagonal anchor moments, /dim.

    For each class c with >= min_count samples in the batch:
      W2^2_c = sum_j (mhat_j - m_c,j)^2 + sum_j (sqrt(shat_j) - sqrt(s_c,j))^2
    Averaged over dim; summed over present classes; mean over #present classes.
    """
    s = anchors.var()                                    # (C, D)
    m = anchors.m                                        # (C, D)
    total = z.new_zeros(())
    n_present = 0
    for c in range(anchors.num_classes):
        mask = (y == c)
        if int(mask.sum()) < min_count:
            continue
        zc = z[mask]
        mhat = zc.mean(0)
        shat = zc.var(0, unbiased=False).clamp_min(1e-8)
        mean_term = (mhat - m[c]).pow(2).sum()
        std_term = (shat.sqrt() - s[c].sqrt()).pow(2).sum()
        total = total + (mean_term + std_term) / anchors.dim
        n_present += 1
    if n_present == 0:
        return z.new_zeros(())
    return total / n_present


def anchor_sep_loss(model: XeniaEmbedModel, n_per_anchor: int = 16) -> torch.Tensor:
    """Sample n_per_anchor points from each anchor (reparam), push through the shared
    head, and require the head to classify them to the right class (CE)."""
    anch = model.anchors
    m, s = anch.m, anch.var()                            # (C, D)
    C, D = anch.num_classes, anch.dim
    eps = torch.randn(C, n_per_anchor, D, device=m.device, dtype=m.dtype)
    samples = m[:, None, :] + s.sqrt()[:, None, :] * eps  # (C, n, D)
    logits = model.head(samples.reshape(C * n_per_anchor, D))
    tgt = torch.arange(C, device=m.device).repeat_interleave(n_per_anchor)
    return F.cross_entropy(logits, tgt)
