import torch
import torch.nn as nn
from typing import Tuple


def normalize_L(L: torch.Tensor, max_norm: float = 10.0, min_norm: float = 1e-3) -> torch.Tensor:
    """Return a scaled copy of L where each (k,k) block is rescaled to lie in [min_norm, max_norm]

    This is a pure (non-mutating) operation to avoid in-place modifications that break autograd.
    """
    norms = torch.linalg.norm(L.reshape(L.size(0), -1), dim=1)
    scale = torch.ones_like(norms, device=L.device)
    upper_mask = norms > max_norm
    lower_mask = norms < min_norm
    if upper_mask.any():
        scale = torch.where(upper_mask, max_norm / norms, scale)
    if lower_mask.any():
        scale = torch.where(lower_mask, min_norm / norms, scale)
    return L * scale.view(-1, 1, 1)


def compute_covariance(L: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute covariance S = L L^T + eps I for a batch of L matrices."""
    eye = torch.eye(L.size(-1), device=L.device).unsqueeze(0)
    return L @ L.transpose(-1, -2) + eps * eye


def sample_gaussian(m: torch.Tensor, L: torch.Tensor, n: int, eps: float = 1e-5) -> torch.Tensor:
    """Sample n points from N(m, L L^T + eps I) using reparameterization.

    m: (k,), L: (k,k)
    returns: (n, k)
    """
    eye = torch.eye(L.size(-1), device=L.device)
    xi = torch.randn(n, L.size(-1), device=m.device)
    return m.unsqueeze(0) + xi @ (L.T + eps * eye)


class AnchorModule(nn.Module):
    """Class-wise Gaussian anchors via (m, L) so S = L Lᵀ + eps I.

    Avoids in-place operations by computing normalized versions of L at
    forward time and never mutating the underlying Parameter directly.
    """
    def __init__(self, num_classes: int, latent_dim: int, eps: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.eps = eps

        self.m = nn.Parameter(torch.zeros(num_classes, latent_dim))
        eye = torch.eye(latent_dim).unsqueeze(0).repeat(num_classes, 1, 1)
        # store L as a parameter; we will use a normalized copy during forward
        self.L = nn.Parameter(0.05 * eye)

    def normalized_L(self, max_norm: float = 10.0, min_norm: float = 1e-3) -> torch.Tensor:
        """Return a normalized copy of self.L (no in-place ops)."""
        return normalize_L(self.L, max_norm=max_norm, min_norm=min_norm)

    def cov(self) -> torch.Tensor:
        Ln = self.normalized_L()
        return compute_covariance(Ln, eps=self.eps)

    def moments(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (m, S) where S = normalized_L @ normalized_L^T + eps I."""
        return self.m, self.cov()

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (m, S, L_norm) for convenience.

        L_norm is the normalized copy of L (useful for sampling without mutating
        parameters).
        """
        Ln = self.normalized_L()
        S = compute_covariance(Ln, eps=self.eps)
        return self.m, S, Ln

    def sample(self, class_idx: int, n: int) -> torch.Tensor:
        m = self.m[class_idx]
        Ln = self.normalized_L()[class_idx]
        return sample_gaussian(m, Ln, n, eps=self.eps)
