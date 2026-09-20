import numpy as np
import torch
from scipy.linalg import sqrtm

from dro_hetero_anchors.src.model.wasserstein import (
    psd_sqrt, gaussian_w2_squared, diagonal_gaussian_w2_squared)


def _rand_psd(k, seed):
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(k, k, generator=g, dtype=torch.float64)
    return B @ B.t() + 1e-3 * torch.eye(k, dtype=torch.float64)


def test_psd_sqrt_is_the_symmetric_principal_root():
    A = _rand_psd(5, 0)
    R = psd_sqrt(A, eps=1e-12)
    assert torch.allclose(R, R.t(), atol=1e-10)             # symmetric (a Cholesky factor is not)
    assert torch.allclose(R @ R, A, atol=1e-8)              # R R = A, not merely R R^T = A
    assert torch.allclose(R, torch.from_numpy(np.real(sqrtm(A.numpy()))), atol=1e-7)


def test_w2_matches_reference_bures_formula():
    m1 = torch.tensor([0.3, -1.0, 2.0, 0.5], dtype=torch.float64)
    m2 = torch.tensor([1.0, 0.2, -0.5, 0.0], dtype=torch.float64)
    S1, S2 = _rand_psd(4, 1), _rand_psd(4, 2)
    r1 = np.real(sqrtm(S1.numpy()))
    ref = float(((m1 - m2) ** 2).sum()) + float(np.trace(
        S1.numpy() + S2.numpy() - 2 * np.real(sqrtm(r1 @ S2.numpy() @ r1))))
    got = float(gaussian_w2_squared(m1, S1, m2, S2, eps=1e-12))
    assert abs(got - ref) < 1e-6
    assert abs(got - float(gaussian_w2_squared(m2, S2, m1, S1, eps=1e-12))) < 1e-6   # symmetric
    assert float(gaussian_w2_squared(m1, S1, m1, S1, eps=1e-12)) < 1e-6              # zero at identity


def test_diagonal_closed_form_agrees_with_full():
    m1, m2 = torch.randn(6, dtype=torch.float64), torch.randn(6, dtype=torch.float64)
    v1, v2 = torch.rand(6, dtype=torch.float64) + 0.1, torch.rand(6, dtype=torch.float64) + 0.1
    full = gaussian_w2_squared(m1, torch.diag(v1), m2, torch.diag(v2), eps=1e-12)
    diag = diagonal_gaussian_w2_squared(m1, v1, m2, v2, eps=1e-12)
    assert torch.allclose(full, diag, atol=1e-8)
