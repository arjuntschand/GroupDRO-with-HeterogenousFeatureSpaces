from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wasserstein import gaussian_w2

def per_class_batch_moments(z: torch.Tensor, y: torch.Tensor, num_classes: int, eps: float) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    out = {}
    eye = torch.eye(z.size(1), device=z.device)
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 2:
            continue
        # Clone to avoid in-place modifications
        zc = z[mask].clone()
        m_hat = zc.mean(dim=0)
        # Avoid in-place subtraction by creating new tensor
        Zc = zc - m_hat.unsqueeze(0)
        # Compute covariance with regularization in one step
        S_hat = (Zc.T @ Zc) / (zc.size(0) - 1) + eps * eye
        out[c] = (m_hat, S_hat)
    return out

def anchor_fit_loss(anchors_m: torch.Tensor, anchors_S: torch.Tensor, batch_moments: Dict[int, Tuple[torch.Tensor, torch.Tensor]], eps: float) -> torch.Tensor:
    losses = []
    for c, (m_hat, S_hat) in batch_moments.items():
        # Validate inputs to W2 distance
        if not torch.isfinite(m_hat).all() or not torch.isfinite(S_hat).all():
            raise ValueError(f"Invalid batch moments for class {c}")
        if not torch.isfinite(anchors_m[c]).all() or not torch.isfinite(anchors_S[c]).all():
            raise ValueError(f"Invalid anchor moments for class {c}")
        
        m_c = anchors_m[c]
        S_c = anchors_S[c]
        w2 = gaussian_w2(m_hat, S_hat, m_c, S_c, eps)
        
        # Validate W2 output
        if not torch.isfinite(w2):
            raise ValueError(f"W2 distance computation failed for class {c}")
            
        losses.append(w2)
        
    if not losses:
        return torch.tensor(0.0, device=anchors_m.device)
    
    loss = torch.mean(torch.stack(losses))
    # Final validation
    if not torch.isfinite(loss):
        raise ValueError("Non-finite anchor fit loss")
    return loss

def anchor_sep_loss(anchors_m: torch.Tensor, anchors_S: torch.Tensor, anchors_L: torch.Tensor, head: nn.Module,
                   num_classes: int, J: int, device: torch.device, sep_method: str = "classifier",
                   margin: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Separation objective for anchors. Two supported methods:
      - "classifier": draw J samples from each anchor (using L) and train the head
                      to predict class labels (cross-entropy). This is the current
                      practical surrogate used in the repo.
      - "w2_margin": compute pairwise Gaussian-W2 distances between anchors and
                     apply a hinge margin loss encouraging anchors to be at least
                     `margin` apart in W2 distance.

    Args:
        anchors_m: (num_classes, k)
        anchors_S: (num_classes, k, k) covariance matrices
        anchors_L: (num_classes, k, k) normalized L matrices for sampling
        sep_method: "classifier" or "w2_margin"
        margin: margin for w2_margin method
    """
    if sep_method == "classifier":
        losses = []
        eye = torch.eye(anchors_L.size(-1), device=device)
        for c in range(num_classes):
            m_c = anchors_m[c]      # (k,)
            L_c = anchors_L[c]      # (k,k)
            xi = torch.randn(J, m_c.size(0), device=device)  # (J,k)
            # Sample via L; avoid mutating parameters. eps can act as small jitter.
            samples = m_c.unsqueeze(0) + xi @ L_c.T
            logits = head(samples)
            target = torch.full((J,), c, dtype=torch.long, device=device)
            ce = F.cross_entropy(logits, target)
            losses.append(ce)
        return torch.mean(torch.stack(losses))
    elif sep_method == "w2_margin":
        # Compute pairwise W2 distances and hinge margin loss
        pairs = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                m_i = anchors_m[i]
                S_i = anchors_S[i]
                m_j = anchors_m[j]
                S_j = anchors_S[j]
                w2 = gaussian_w2(m_i, S_i, m_j, S_j, eps=eps)
                # w2 is scalar tensor
                hinge = torch.clamp(margin - w2, min=0.0)
                pairs.append(hinge)
        if not pairs:
            return torch.tensor(0.0, device=anchors_m.device)
        return torch.mean(torch.stack(pairs))
    else:
        raise ValueError(f"Unknown sep_method: {sep_method}")
