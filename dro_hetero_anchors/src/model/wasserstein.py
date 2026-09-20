"""Squared 2-Wasserstein distance between Gaussians.

History (2026-09-20). psd_sqrt() used to return a CHOLESKY factor, not the symmetric principal
square root the Bures-Wasserstein formula needs, and stabilize_matrix() added a ridge scaled by
the Frobenius norm, which changes the quantity being optimised. For diagonal matrices the two
square roots coincide, so diagonal-anchor runs were unaffected; every full-covariance run before
this date trained against a covariance term that was not the Bures term. The diagonal closed form
is now the main path, and the full-covariance path uses an eigendecomposition.
"""
import torch


def diagonal_gaussian_w2_squared(m1, v1, m2, v2, eps: float = 1e-7) -> torch.Tensor:
    """W2^2 between N(m1, diag v1) and N(m2, diag v2): |m1-m2|^2 + |sqrt(v1)-sqrt(v2)|^2."""
    mean_term = (m1 - m2).square().sum(dim=-1)
    cov_term = (torch.sqrt(v1.clamp_min(eps)) - torch.sqrt(v2.clamp_min(eps))).square().sum(dim=-1)
    return mean_term + cov_term


def psd_sqrt(mat: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Symmetric principal square root of a PSD matrix, by eigendecomposition."""
    if torch.isnan(mat).any():
        raise ValueError("Input matrix contains NaN values")
    mat = 0.5 * (mat + mat.transpose(-2, -1))
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp_min(eps)
    return eigvecs @ torch.diag_embed(torch.sqrt(eigvals)) @ eigvecs.transpose(-2, -1)


def gaussian_w2_mean_term(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    return torch.sum((m1 - m2) ** 2, dim=-1)


def gaussian_w2_bures_term(S1: torch.Tensor, S2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """tr(S1 + S2 - 2 (S1^{1/2} S2 S1^{1/2})^{1/2}), with principal square roots."""
    S1_sqrt = psd_sqrt(S1, eps)
    inner_sqrt = psd_sqrt(S1_sqrt @ S2 @ S1_sqrt, eps)
    return torch.einsum('...ii->...', S1 + S2 - 2.0 * inner_sqrt).clamp_min(0.0)


def gaussian_w2_squared(m1, S1, m2, S2, eps: float = 1e-7) -> torch.Tensor:
    """Squared Bures-Wasserstein distance between N(m1, S1) and N(m2, S2), full covariances."""
    return gaussian_w2_mean_term(m1, m2) + gaussian_w2_bures_term(S1, S2, eps)


# Backwards-compatible name. It always returned the SQUARED distance.
gaussian_w2 = gaussian_w2_squared
