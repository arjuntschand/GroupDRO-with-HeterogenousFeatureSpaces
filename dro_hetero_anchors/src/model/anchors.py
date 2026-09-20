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
    xi = torch.randn(n, L.size(-1), device=m.device)
    xi2 = torch.randn(n, L.size(-1), device=m.device)
    return m.unsqueeze(0) + xi @ L.T + (eps ** 0.5) * xi2


class AnchorModule(nn.Module):
    """Class-wise Gaussian anchors mu_c = N(m_c, Sigma_c).

    diagonal=True (the paper's parameterisation, eq. 6): one scale parameter per class and
    dimension, variance = softplus(raw_scale) + eps. The SAME variance defines the covariance
    the alignment loss compares against and the distribution the separation loss samples from,
    so the two can never disagree, and variances are positive by construction.

    diagonal=False (legacy, kept to reproduce earlier runs): full covariance S = L L^T + eps I
    from a free factor L. Before 2026-09-20 the moments used L L^T + eps I while sample_gaussian
    drew from xi (L^T + eps I) and the separation loss from xi L^T: three different
    distributions. sample_gaussian now draws from N(m, L L^T + eps I) exactly.
    """
    def __init__(self, num_classes: int, latent_dim: int, eps: float = 1e-5, diagonal: bool = False):
        super().__init__()
        self.diagonal = diagonal
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.eps = eps

        self.m = nn.Parameter(torch.zeros(num_classes, latent_dim))
        if diagonal:
            self.raw_scale = nn.Parameter(torch.full((num_classes, latent_dim), -3.0))
        else:
            eye = torch.eye(latent_dim).unsqueeze(0).repeat(num_classes, 1, 1)
            self.L = nn.Parameter(0.05 * eye)

    def variance(self) -> torch.Tensor:
        """(C, D) anchor variances; diagonal mode only."""
        return torch.nn.functional.softplus(self.raw_scale) + self.eps

    def normalized_L(self, max_norm: float = 10.0, min_norm: float = 1e-3) -> torch.Tensor:
        if self.diagonal:
            return torch.diag_embed(torch.sqrt(self.variance()))
        return normalize_L(self.L, max_norm=max_norm, min_norm=min_norm)

    def cov(self) -> torch.Tensor:
        if self.diagonal:
            return torch.diag_embed(self.variance())
        return compute_covariance(self.normalized_L(), eps=self.eps)

    def moments(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.m, self.cov()

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (m, S, F) with F a factor such that m + xi F^T ~ N(m, S).

        Diagonal mode: S = diag(variance), F = diag(std), exactly consistent. Legacy mode:
        S = L L^T + eps I and F = L (the eps I term, 1e-5, is not in the factor)."""
        if self.diagonal:
            var = self.variance()
            return self.m, torch.diag_embed(var), torch.diag_embed(torch.sqrt(var))
        Ln = self.normalized_L()
        return self.m, compute_covariance(Ln, eps=self.eps), Ln

    def sample(self, class_idx: int, n: int) -> torch.Tensor:
        m = self.m[class_idx]
        if self.diagonal:
            std = torch.sqrt(self.variance()[class_idx])
            return m.unsqueeze(0) + torch.randn(n, self.latent_dim, device=m.device) * std
        return sample_gaussian(m, self.normalized_L()[class_idx], n, eps=self.eps)
