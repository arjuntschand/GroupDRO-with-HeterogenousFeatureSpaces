import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

@dataclass
class GroupStats:
    """Track per-group statistics for GroupDRO."""
    losses: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)  # q_g history
    counts: List[int] = field(default_factory=list)     # samples per group per batch
    per_class_correct: Dict[int, List[int]] = field(default_factory=dict)  # per-class correct counts
    per_class_total: Dict[int, List[int]] = field(default_factory=dict)    # per-class total counts

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
    4. KL divergence penalty to prevent diverging too far from initial distribution π:
        loss += kl_lambda * KL(q || π)
       This regularizes the learned weights toward the natural group distribution.

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
                 gamma: float = 1.0,
                 group_counts: Optional[List[int]] = None,
                 kl_lambda: float = 0.0,
                 uniform_init: bool = False,
                 use_regret: bool = False,
                 optimal_losses: Optional[List[float]] = None,
                 ema_decay: float = 0.0,
                 update_every: int = 1):
        self.num_groups = num_groups
        self.eta = eta
        self.device = device or torch.device('cpu')
        self.update_mode = update_mode  # 'exp'|'softmax'|'exp_smooth'
        self.robust_objective = robust_objective  # 'weighted'|'max'|'logsumexp'
        self.gamma = gamma  # smoothing factor for exp_smooth
        self.kl_lambda = kl_lambda  # coefficient for KL(q || π) penalty
        self.uniform_init = uniform_init  # whether to initialize q uniformly
        # Regret-DRO: reweight by regret R_g = max(0, L_g - L*_g) instead of raw loss,
        # where L*_g is each group's optimal loss (from a per-group "personal" model).
        # Groups at their achievable floor get ~0 regret -> stop being upweighted.
        self.use_regret = use_regret
        # EMBED_Experiments_Description.docx, "The training loop":
        #     ema_loss[g] = DECAY * ema_loss[g] + (1 - DECAY) * L[g]        DECAY = 0.9
        #     every N steps:  excess = max(0, ema_loss - R*);  lambda *= exp(gamma * excess)
        # The EMA exists so a group absent from most batches does not update lambda from stale
        # information, and the N-step cadence keeps a single noisy batch from moving lambda.
        # Neither existed here: the tabular path recomputed q from the CURRENT batch every step.
        # ema_decay = 0 keeps the old behaviour, so nothing changes unless a config asks for it.
        self.ema_decay = ema_decay
        self.update_every = max(1, int(update_every))
        self._step = 0
        self.ema = None
        self.set_optimal_losses(optimal_losses)

        # Initialize reference π based on group distribution (for KL penalty)
        # This is always proportional to group sizes for KL regularization
        if group_counts is not None and len(group_counts) == num_groups:
            total = sum(group_counts)
            if total > 0:
                pi = torch.tensor([c / total for c in group_counts], 
                                  device=self.device, dtype=torch.float32)
            else:
                pi = torch.ones(num_groups, device=self.device) / num_groups
        else:
            pi = torch.ones(num_groups, device=self.device) / num_groups
        
        # Store π as the reference distribution for KL penalty
        self.pi = pi.detach()
        
        # Initialize q: uniform if uniform_init=True, else proportional to group sizes
        if uniform_init:
            self.q = torch.ones(num_groups, device=self.device) / num_groups
        else:
            self.q = pi.clone().detach()

        # track statistics for analysis
        self.group_stats = {i: GroupStats() for i in range(num_groups)}
        # Track the best observed worst-group accuracy across training. Initialize
        # to -inf so the first observed value replaces it via max().
        self.worst_group_acc = float("-inf")
        
        # Track KL penalty history
        self.kl_penalty_history: List[float] = []
        
        # Last batch statistics for logging
        self._last_group_losses: Dict[int, torch.Tensor] = {}
        self._last_group_accs: Dict[int, float] = {}
        self._last_group_counts: Dict[int, int] = {}
        self._last_per_class_accs: Dict[int, Dict[int, float]] = {}  # group -> class -> acc
        self._last_kl_penalty: float = 0.0
    
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

    def compute_kl_divergence(self) -> torch.Tensor:
        """Compute KL(q || π) where π is the initial/reference distribution.
        
        KL(q || π) = sum_g q_g * log(q_g / π_g)
        
        Returns scalar tensor for adding to the loss.
        """
        eps = 1e-8
        q_safe = self.q.clamp(min=eps)
        pi_safe = self.pi.clamp(min=eps)
        kl = (q_safe * (q_safe.log() - pi_safe.log())).sum()
        return kl

    def set_optimal_losses(self, optimal_losses):
        """Set per-group optimal (achievable-best) losses L*_g for Regret-DRO.
        optimal_losses: list/dict of length num_groups, or None to disable regret."""
        if optimal_losses is None:
            self.optimal = None
            return
        if isinstance(optimal_losses, dict):
            vals = [float(optimal_losses.get(g, 0.0)) for g in range(self.num_groups)]
        else:
            vals = [float(x) for x in optimal_losses]
        self.optimal = torch.tensor(vals, device=self.device, dtype=torch.float32)

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

            # EMA over observed losses, updated only for groups present this batch, so an
            # absent group keeps its last estimate rather than being driven to zero.
            self._step += 1
            if self.ema_decay > 0:
                if self.ema is None:
                    init = self.optimal.clone() if self.optimal is not None \
                        else losses_tensor.clone()
                    self.ema = init.to(self.device)
                d = self.ema_decay
                upd = d * self.ema + (1 - d) * losses_tensor
                self.ema = torch.where(active_mask_t, upd, self.ema)
                losses_tensor = self.ema.clone()
                # only act every N steps; in between, lambda is left exactly as it was
                if self._step % self.update_every != 0:
                    return

            # Regret-DRO: drive the q-update by regret R_g = max(0, L_g - L*_g)
            # instead of raw loss, so groups at their optimum stop being upweighted.
            if self.use_regret and self.optimal is not None:
                losses_tensor = torch.clamp(losses_tensor - self.optimal, min=0.0)

            if self.update_mode == 'exp':
                # MWU on all entries; absent groups keep neutral multiplier (exp(eta*0)=1).
                #
                # The exponent is clamped for the reason train_embed_xenia documents at its
                # line 313: unclamped it reached ~797 there, overflowed to inf, and lambda went
                # exactly one-hot so the model trained on a single group and scored 9.7%. That
                # runaway is what motivated the tabular configs to switch to stateless softmax,
                # which then erased lambda whenever regret's clamp zeroed every group. Guarding
                # the exponent keeps MWU's accumulation without the divergence.
                q_new = self.q * torch.exp(torch.clamp(self.eta * losses_tensor, max=20.0))
                q_new = torch.clamp(q_new, min=1e-8)
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

            self.q.copy_(q_new)
    
    def forward(self, logits: torch.Tensor, y: torch.Tensor,
                g: torch.Tensor, num_classes: Optional[int] = None,
                class_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute GroupDRO weighted loss and update statistics.
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            y: Ground truth labels (batch_size,)
            g: Group labels (batch_size,)
            num_classes: Number of classes (inferred from logits if not provided)
            
        Returns:
            Weighted average of per-group losses (with optional KL penalty).
        """
        if num_classes is None:
            num_classes = logits.shape[1]
            
        group_losses = {}
        group_accs = {}
        group_counts = {}
        per_class_accs = {}  # group -> class -> accuracy
        
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
                loss = nn.functional.cross_entropy(g_logits, g_y, weight=class_weight)
                pred = g_logits.argmax(dim=1)
                acc = (pred == g_y).float().mean().item()
                
                group_losses[gid] = loss
                group_accs[gid] = acc
                
                # Compute per-class accuracy for this group
                per_class_accs[gid] = {}
                for cid in range(num_classes):
                    class_mask = (g_y == cid)
                    class_count = class_mask.sum().item()
                    if class_count > 0:
                        class_correct = ((pred == g_y) & class_mask).sum().item()
                        per_class_accs[gid][cid] = class_correct / class_count
                    else:
                        per_class_accs[gid][cid] = None  # No samples of this class
                
                weighted_loss = weighted_loss + self.q[gid] * loss
                losses_list.append(loss)
                present_losses.append(loss)
                max_loss = torch.maximum(max_loss, loss.detach())
            else:
                # absent group: still append a zero loss for consistent shapes in logsumexp
                losses_list.append(torch.tensor(0.0, device=self.device))
                per_class_accs[gid] = {cid: None for cid in range(num_classes)}
        
        # track statistics
        for gid in range(self.num_groups):
            if gid in group_losses:
                stats = self.group_stats[gid]
                stats.losses.append(group_losses[gid].item())
                stats.accuracies.append(group_accs[gid])
                stats.weights.append(self.q[gid].item())
                stats.counts.append(group_counts[gid])

        # Store last batch statistics for deferred weight update and logging.
        # We must NOT update self.q here because that would mutate a tensor
        # that was used in the forward pass before the backward call and
        # break autograd. The trainer should call update_weights(...) after
        # loss.backward() / optimizer.step().
        self._last_group_losses = group_losses
        self._last_group_counts = group_counts
        self._last_group_accs = group_accs
        self._last_per_class_accs = per_class_accs

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

        # Add KL divergence penalty: kl_lambda * KL(q || π)
        # This regularizes learned weights toward the natural group distribution
        if self.kl_lambda > 0:
            kl_penalty = self.compute_kl_divergence()
            final_loss = final_loss + self.kl_lambda * kl_penalty
            self._last_kl_penalty = kl_penalty.item()
            self.kl_penalty_history.append(self._last_kl_penalty)
        else:
            self._last_kl_penalty = 0.0

        return final_loss
    
    def get_last_batch_stats(self) -> Dict[str, Any]:
        """Get statistics from the last forward pass for logging.
        
        Returns:
            Dictionary with:
                - group_losses: Dict[int, float]
                - group_accs: Dict[int, float]
                - group_counts: Dict[int, int]
                - per_class_accs: Dict[int, Dict[int, float]]
                - weights: List[float]
                - kl_penalty: float
        """
        return {
            "group_losses": {gid: loss.item() if hasattr(loss, 'item') else loss 
                           for gid, loss in self._last_group_losses.items()},
            "group_accs": self._last_group_accs.copy(),
            "group_counts": self._last_group_counts.copy(),
            "per_class_accs": self._last_per_class_accs.copy(),
            "weights": self.q.detach().cpu().tolist(),
            "kl_penalty": self._last_kl_penalty,
            "pi": self.pi.detach().cpu().tolist(),
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get GroupDRO configuration for logging.
        
        Returns:
            Dictionary with all configuration parameters.
        """
        return {
            "num_groups": self.num_groups,
            "eta": self.eta,
            "update_mode": self.update_mode,
            "robust_objective": self.robust_objective,
            "gamma": self.gamma,
            "kl_lambda": self.kl_lambda,
            "initial_pi": self.pi.detach().cpu().tolist(),
        }
    
    def reset_stats(self):
        """Reset all statistics (useful between epochs)."""
        for gid in range(self.num_groups):
            self.group_stats[gid] = GroupStats()
        self.kl_penalty_history = []
        self.worst_group_acc = float("-inf")