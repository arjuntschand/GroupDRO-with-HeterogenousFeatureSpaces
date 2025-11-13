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
    """
    Implements Group Distributionally Robust Optimization with multiplicative weights.
    
    Following "Distributionally Robust Neural Networks for Group Shifts: On the Importance 
    of Groups for DRO" (Sagawa et al. 2020), but adapted for heterogeneous feature spaces.
    """
    def __init__(self, num_groups: int, eta: float = 0.1, device: torch.device = None):
        self.num_groups = num_groups
        self.eta = eta
        self.device = device or torch.device('cpu')
        
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
        """
        Update q_g weights via multiplicative weights:
        q_g <- q_g * exp(eta * loss_g)
        
        Uses non-in-place operations to avoid autograd issues.
        """
        with torch.no_grad():
            # Create new weights without modifying old ones
            q_new = self.q.clone()
            
            # Update active groups
            for gid in range(self.num_groups):
                if group_counts.get(gid, 0) > 0:
                    q_new[gid] = self.q[gid] * torch.exp(self.eta * group_losses[gid].detach())
            
            # Safe renormalization
            if q_new.sum() > 0:
                q_new = q_new / q_new.sum()
            
            # Only after all computations, update self.q
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
        
        # Compute weighted loss for each group
        weighted_loss = torch.tensor(0.0, device=self.device)
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
                
                # Accumulate weighted loss directly (no list/append needed)
                weighted_loss = weighted_loss + self.q[gid] * loss
        
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

        return weighted_loss