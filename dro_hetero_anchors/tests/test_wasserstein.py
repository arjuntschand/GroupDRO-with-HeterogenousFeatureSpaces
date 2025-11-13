import torch
from src.model.wasserstein import psd_sqrt, gaussian_w2


def test_psd_sqrt_reconstruct():
    torch.manual_seed(0)
    k = 4
    B = torch.randn(k, k)
    A = B @ B.t() + 1e-3 * torch.eye(k)
    L = psd_sqrt(A, eps=1e-6)
    recon = L @ L.t()
    assert torch.allclose(A, recon, atol=1e-3, rtol=1e-3)


def test_gaussian_w2_symmetry_nonneg():
    torch.manual_seed(1)
    k = 3
    m1 = torch.randn(k)
    m2 = torch.randn(k)
    B1 = torch.randn(k, k)
    B2 = torch.randn(k, k)
    S1 = B1 @ B1.t() + 1e-3 * torch.eye(k)
    S2 = B2 @ B2.t() + 1e-3 * torch.eye(k)
    w12 = gaussian_w2(m1, S1, m2, S2, eps=1e-6)
    w21 = gaussian_w2(m2, S2, m1, S1, eps=1e-6)
    assert w12 >= 0
    assert torch.allclose(w12, w21, atol=1e-5)
