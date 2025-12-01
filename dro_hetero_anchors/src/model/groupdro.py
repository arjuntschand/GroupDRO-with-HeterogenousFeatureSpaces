import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass

@dataclass
class GroupStats:
    """Track per-group statistics for GroupDRO."""
    losses: List[float]
    accuracies: List[float]
    weights: List[float]  # q_g history
    counts: List[int]     # samples per group per batch

class GroupDRO:
    """Flexible Group Distributionally Robust Optimization module.

    Original (baseline) behavior: multiplicative weights update (MWU)
        q_g <- q_g * exp(eta * loss_g); normalize.

    Extensions added (based on new project formulation PDF):
    1. Smoothing / momentum on weights (gamma parameter):
        q_new = (1-gamma) * q_old + gamma * softmax(eta * losses)
       When gamma=1 this reduces to a pure softmax over exponentiated losses.
    2. Alternative objective modes:
        - 'weighted': returns sum_g q_g * loss_g (standard DRO surrogate)
        - 'max': returns max_g loss_g (worst-group empirical risk)
        - 'logsumexp': temperature-controlled smooth max: (1/eta) * log(sum exp(eta*loss_g))
          (uses same eta as update temperature; stable for small batch sizes)
    3. Update modes:
        - 'exp' (baseline MWU)
        - 'softmax' (direct projection to softmax of scaled losses)
        - 'exp_smooth' (MWU then convex combination with previous weights via gamma)

    Design goals:
      * Non-in-place modifications until assignment to self.q
      * Detach weights from autograd (dual optimization via closed-form updates)
      * Track per-group statistics for post-hoc analysis.
    """
    def __init__(self,
                 num_groups: int,
                 eta: float = 0.1,
                 device: torch.device = None,
                 update_mode: str = "exp",
                 robust_objective: str = "weighted",
                 gamma: float = 1.0):
        self.num_groups = num_groups
        self.eta = eta
        self.device = device or torch.device('cpu')
        self.update_mode = update_mode  # 'exp'|'softmax'|'exp_smooth'
        self.robust_objective = robust_objective  # 'weighted'|'max'|'logsumexp'
        self.gamma = gamma  # smoothing factor for exp_smooth

        # initialize uniform group weights (no gradients needed)
        self.q = (torch.ones(num_groups, device=self.device) / num_groups).detach()

        # track statistics for analysis
        self.group_stats = {i: GroupStats([], [], [], []) for i in range(num_groups)}
        # Track the best observed worst-group accuracy across training. Initialize
        # to -inf so the first observed value replaces it via max().
        self.worst_group_acc = float("-inf")
    
    def compute_group_losses(self, logits: torch.Tensor, y: torch.Tensor, 
                           g: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Compute per-group cross entropy losses."""
        losses = {}
        for gid in range(self.num_groups):
            mask = (g == gid)
            if mask.sum() > 0:
                losses[gid] = nn.functional.cross_entropy(logits[mask], y[mask])
            else:
                losses[gid] = torch.tensor(0.0, device=self.device)
        return losses

    def update_weights(self, group_losses: Dict[int, torch.Tensor],
                       group_counts: Dict[int, int]):
        """Update group weights according to selected mode.

        Modes:
          exp:        multiplicative weights (legacy)
          softmax:    q <- softmax(eta * losses)
          exp_smooth: q_mwu then q <- (1-gamma) * q_old + gamma * q_mwu
        """
        with torch.no_grad():
            # Build tensors of losses and an active mask (groups seen this batch)
            active_losses = []
            active_mask = []
            for gid in range(self.num_groups):
                is_active = group_counts.get(gid, 0) > 0
                active_mask.append(is_active)
                if is_active:
                    active_losses.append(group_losses[gid].detach())
                else:
                    # placeholder; won’t be used where masked
                    active_losses.append(torch.tensor(0.0, device=self.device))
            losses_tensor = torch.stack(active_losses)  # shape [G]
            active_mask_t = torch.tensor(active_mask, dtype=torch.bool, device=self.device)

            if self.update_mode == 'exp':
                # MWU on all entries; absent groups keep neutral multiplier (exp(eta*0)=1)
                q_new = self.q * torch.exp(self.eta * losses_tensor)
            elif self.update_mode == 'softmax':
                # Softmax only over active groups; absent get ~0 weight
                scaled = self.eta * losses_tensor
                # mask absent groups with -inf so they get zero after softmax
                scaled = torch.where(active_mask_t, scaled, torch.tensor(float('-inf'), device=self.device))
                q_new = torch.softmax(scaled, dim=0)
                # replace NaNs if all groups absent (shouldn’t happen) with uniform
                if torch.isnan(q_new).any():
                    q_new = torch.ones_like(self.q) / self.num_groups
            elif self.update_mode == 'exp_smooth':
                q_mwu = self.q * torch.exp(self.eta * losses_tensor)
                if q_mwu.sum() > 0:
                    q_mwu = q_mwu / q_mwu.sum()
                # convex combination
                q_new = (1 - self.gamma) * self.q + self.gamma * q_mwu
            else:
                # fallback: keep weights uniform
                q_new = torch.ones_like(self.q) / self.num_groups

            # normalize (except when already normalized by softmax)
            if self.update_mode not in ('softmax') and q_new.sum() > 0:
                q_new = q_new / q_new.sum()

            # No min_q projection: follow PDF strictly (smoothing only)

            self.q.copy_(q_new)
    
    def forward(self, logits: torch.Tensor, y: torch.Tensor, 
                g: torch.Tensor) -> torch.Tensor:
        """
        Compute GroupDRO weighted loss and update statistics.
        Returns weighted average of per-group losses.
        """
        group_losses = {}
        group_accs = {}
        group_counts = {}
        
        # Compute weighted components
        weighted_loss = torch.tensor(0.0, device=self.device)
        max_loss = torch.tensor(float('-inf'), device=self.device)
        losses_list = []
        present_losses = []
        for gid in range(self.num_groups):
            mask = (g == gid)
            count = mask.sum().item()
            group_counts[gid] = count
            
            if count > 0:
                g_logits = logits[mask]
                g_y = y[mask]
                
                # compute loss and accuracy for this group
                loss = nn.functional.cross_entropy(g_logits, g_y)
                pred = g_logits.argmax(dim=1)
                acc = (pred == g_y).float().mean().item()
                
                group_losses[gid] = loss
                group_accs[gid] = acc
                
                weighted_loss = weighted_loss + self.q[gid] * loss
                losses_list.append(loss)
                present_losses.append(loss)
                max_loss = torch.maximum(max_loss, loss.detach())
            else:
                # absent group: still append a zero loss for consistent shapes in logsumexp
                losses_list.append(torch.tensor(0.0, device=self.device))
        
        # track statistics
        for gid in range(self.num_groups):
            if gid in group_losses:
                stats = self.group_stats[gid]
                stats.losses.append(group_losses[gid].item())
                stats.accuracies.append(group_accs[gid])
                stats.weights.append(self.q[gid].item())
                stats.counts.append(group_counts[gid])

        # Store last batch losses/counts for deferred weight update.
        # We must NOT update self.q here because that would mutate a tensor
        # that was used in the forward pass before the backward call and
        # break autograd. The trainer should call update_weights(...) after
        # loss.backward() / optimizer.step().
        self._last_group_losses = group_losses
        self._last_group_counts = group_counts

        # track worst group accuracy
        if len(group_accs) > 0:
            worst_acc = min(group_accs.values())
            # We want to track the best (largest) worst-group accuracy seen so far.
            self.worst_group_acc = max(self.worst_group_acc, worst_acc)

        # Select objective variant
        if self.robust_objective == 'weighted':
            final_loss = weighted_loss
        elif self.robust_objective == 'max':
            final_loss = max_loss
        elif self.robust_objective == 'logsumexp':
            # Smooth approximation to max over present groups only
            if len(present_losses) == 0:
                final_loss = weighted_loss  # fallback
            else:
                losses_stack = torch.stack(present_losses)
                final_loss = (1.0 / max(self.eta, 1e-8)) * torch.log(torch.exp(self.eta * losses_stack).sum())
        else:
            final_loss = weighted_loss

        return final_loss