from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wasserstein import gaussian_w2


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and focusing on hard examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    When gamma > 0, reduces the loss for well-classified examples and focuses
    on hard, misclassified examples. This can help GroupDRO by naturally
    upweighting difficult samples.
    
    Args:
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard examples.
        alpha: Class weights (optional). Can be scalar or per-class tensor.
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Cross entropy with label smoothing.
    
    Smoothed targets: y_smooth = (1 - smoothing) * y_onehot + smoothing / num_classes
    
    This regularization can prevent overconfidence and help generalization.
    
    Args:
        smoothing: Label smoothing factor (default 0.1)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = (-smooth_targets * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

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


@torch.no_grad()
def group_alignment_losses(encoders, anchors, loader, device, num_groups: int, num_classes: int,
                           eps: float, feature_indices=None, diagonal: bool = False):
    """Per-group alignment loss L^align_g on a held-out loader: W2 between each group's
    per-class latent moments and the class anchors. Algorithm 1 tracks L_g + alpha * L^align_g
    in the running loss that drives lambda; this supplies the second term. Leaves the encoders
    in eval mode, like evaluate()."""
    for e in encoders.values():
        e.eval()
    zs = {g: [] for g in range(num_groups)}
    ys = {g: [] for g in range(num_groups)}
    for x, y, g in loader:
        x, y, g = x.to(device), y.to(device), g.to(device)
        for gid, enc in encoders.items():
            m = (g == gid)
            if m.sum() == 0:
                continue
            xg = x[m]
            if feature_indices is not None and gid in feature_indices:
                xg = xg[:, feature_indices[gid]]
            zs[gid].append(enc(xg)); ys[gid].append(y[m])
    m_anc, S_anc, _ = anchors.forward()
    out = []
    for gid in range(num_groups):
        if not zs[gid]:
            out.append(0.0); continue
        mom = per_class_batch_moments(torch.cat(zs[gid]), torch.cat(ys[gid]), num_classes, eps)
        if diagonal and mom:
            mom = {c: (m, torch.diag(torch.diagonal(S))) for c, (m, S) in mom.items()}
        out.append(float(anchor_fit_loss(m_anc, S_anc, mom, eps)) if mom else 0.0)
    return out


def diagonalize_moments(moments):
    """Keep only the diagonal of each class covariance (diagonal-anchor variant)."""
    return {c: (m, torch.diag(torch.diagonal(S))) for c, (m, S) in moments.items()}
