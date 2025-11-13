import torch

def stabilize_matrix(mat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Return a copy of mat that is symmetric and regularized."""
    # Force symmetry first
    mat_sym = 0.5 * (mat + mat.transpose(-2, -1))
    # Add adaptive regularization based on matrix norm
    eye = torch.eye(mat.shape[-1], device=mat.device)
    frob_norm = torch.norm(mat_sym)
    reg_eps = max(eps, eps * frob_norm)
    return mat_sym + reg_eps * eye

def psd_sqrt(mat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute square root of positive semidefinite matrix.
    
    Returns a copy - no in-place modifications.
    """
    # Validate input
    if torch.isnan(mat).any():
        raise ValueError("Input matrix contains NaN values")
    
    # Get a stabilized copy for decomposition
    mat_stable = stabilize_matrix(mat, eps)
    
    # Use Cholesky when we can (fastest)
    try:
        L = torch.linalg.cholesky(mat_stable)
        return L
    except RuntimeError:
        pass
    
    # Fall back to eigendecomposition with careful conditioning
    eigvals, eigvecs = torch.linalg.eigh(mat_stable)
    # Scale minimum eigenvalue relative to maximum for better conditioning
    min_eigval = eps * torch.max(eigvals.real)
    sqrt_vals = torch.sqrt(torch.clamp(eigvals.real, min=min_eigval))
    
    # Reconstruct sqrt carefully to maintain symmetry
    sqrt_mat = (eigvecs * sqrt_vals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
    sqrt_mat = sqrt_mat.real
    sqrt_mat = 0.5 * (sqrt_mat + sqrt_mat.transpose(-2, -1))
    
    if torch.isnan(sqrt_mat).any():
        raise ValueError("Matrix square root computation failed")
    return sqrt_mat

def gaussian_w2_mean_term(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    return torch.sum((m1 - m2) ** 2, dim=-1)

def gaussian_w2_bures_term(S1: torch.Tensor, S2: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute Bures-Wasserstein coupling term tr(S1 + S2 - 2(S1^{1/2}S2S1^{1/2})^{1/2}).
    
    Matrices S1 and S2 must be positive definite.
    """
    # Get matrix square roots without modifying inputs
    S1_sqrt = psd_sqrt(S1, eps)
    inner = S1_sqrt @ S2 @ S1_sqrt.transpose(-2, -1)
    inner_sqrt = psd_sqrt(inner, eps)
    return torch.einsum('...ii->...', S1 + S2 - 2.0 * inner_sqrt)

def gaussian_w2(m1: torch.Tensor, S1: torch.Tensor, m2: torch.Tensor, S2: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return gaussian_w2_mean_term(m1, m2) + gaussian_w2_bures_term(S1, S2, eps)
