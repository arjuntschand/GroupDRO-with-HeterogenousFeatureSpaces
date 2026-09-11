"""Baselines from the REMIND paper (arXiv 2603.00046), adapted to our datasets.

Three methods, requested as additional comparisons:

  Reweigh   a long-tailed learning baseline. Fixed per-group loss weights proportional to
            1/n_g, normalised to the simplex. Unlike GroupDRO these weights do not adapt
            during training; small groups simply count for more from the start.

  FlexMoE   a multi-modal baseline (Yun et al., cited as [42] in the REMIND paper). Rather
            than zero-padding absent modalities, it learns a per-group embedding B_k that
            stands in for whatever that group is missing, so every sample presents a full
            modality set to a shared fusion block. Fusion here is a Soft MoE.

  REMIND    their method: the same Soft MoE fusion plus a distributionally robust outer
            loop, with the group weights lambda refreshed every N steps from per-group loss.

IMPORTANT, read before quoting any of these numbers. The REMIND paper does not include a
code-availability statement and we could not locate a public repository, so these are
reimplementations from the paper's description, not the authors' released code. The
experiment spec asks for "as published, use their released code". Until that code is
obtained these should be reported as "our reimplementation" and treated as indicative. The
most likely source of divergence is the Soft MoE configuration (expert count, token length)
which the paper specifies only partially.

Adapting the modality framing to tabular data
---------------------------------------------
REMIND and FlexMoE are written for modality combinations. Our tabular datasets have the same
structure under a different name: a group is defined by which block of features it has. On
NHANES the blocks are survey (10 features, everyone), exam (3, G1 and G2) and labs (7, G2
only), so the groups are exactly {survey}, {survey, exam}, {survey, exam, labs}. Fed-Heart is
handled the same way, with one block per distinct feature subset. This makes the mapping to
"modality combination groups" exact rather than analogical.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Reweigh ──────────────────────────────────────────────────────────────────────────────

def inverse_frequency_weights(group_counts: Sequence[int], device: torch.device,
                              power: float = 1.0) -> torch.Tensor:
    """Fixed simplex weights proportional to (1 / n_g) ** power.

    power=1 is plain inverse frequency. The REMIND paper's long-tailed baselines use this
    family; we keep power configurable so the strength can be swept if needed.
    """
    n = torch.tensor([max(1, int(c)) for c in group_counts], dtype=torch.float32, device=device)
    w = (1.0 / n) ** power
    return w / w.sum()


class ReweighLoss(nn.Module):
    """Per-group cross-entropy combined with FIXED inverse-frequency weights.

    Deliberately not adaptive: that is what separates it from GroupDRO. If this matches
    GroupDRO closely on a dataset, it means the adaptivity is not buying anything there.
    """

    def __init__(self, num_groups: int, group_counts: Sequence[int], device: torch.device,
                 power: float = 1.0, class_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_groups = num_groups
        self.register_buffer("w", inverse_frequency_weights(group_counts, device, power))
        self.class_weight = class_weight
        self.last_group_losses: Dict[int, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        total = logits.new_zeros(())
        seen = logits.new_zeros(())
        self.last_group_losses = {}
        for gid in range(self.num_groups):
            m = (g == gid)
            if not bool(m.any()):
                continue
            li = F.cross_entropy(logits[m], y[m], weight=self.class_weight)
            self.last_group_losses[gid] = li.detach()
            total = total + self.w[gid] * li
            seen = seen + self.w[gid]
        # renormalise over groups actually present, so batches missing a group are not
        # silently down-weighted overall
        return total / seen.clamp_min(1e-8)


# ── Soft MoE fusion, shared by FlexMoE and REMIND ────────────────────────────────────────

class SoftMoE(nn.Module):
    """Soft MoE fusion block (Puigcerver et al., as used by REMIND).

    Tokens are dispatched to experts by a learnable routing matrix Phi. Dispatch weights are
    a column-wise softmax over tokens, combine weights a row-wise softmax over experts, so
    every expert sees a weighted average of all tokens rather than a hard assignment.
    """

    def __init__(self, dim: int, n_experts: int = 4, expert_hidden: int = 64,
                 n_slots: int = 1):
        super().__init__()
        self.phi = nn.Parameter(torch.randn(dim, n_experts * n_slots) * 0.02)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, expert_hidden), nn.GELU(),
                          nn.Linear(expert_hidden, dim))
            for _ in range(n_experts)])
        self.n_experts, self.n_slots = n_experts, n_slots

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, D) tokens -> (B, D) fused representation."""
        logits = torch.einsum("btd,dk->btk", z, self.phi)        # (B, T, E*S)
        dispatch = logits.softmax(dim=1)                          # over tokens
        combine = logits.flatten(1).softmax(dim=-1).view_as(logits)  # over slots
        slots = torch.einsum("btk,btd->bkd", dispatch, z)         # (B, E*S, D)
        slots = slots.view(z.size(0), self.n_experts, self.n_slots, -1)
        out = torch.stack([self.experts[e](slots[:, e]) for e in range(self.n_experts)], 1)
        out = out.view(z.size(0), self.n_experts * self.n_slots, -1)
        return torch.einsum("btk,bkd->bd", combine, out)


class FlexMoEModel(nn.Module):
    """Shared per-block projections + learnable group embeddings + Soft MoE fusion.

    The distinguishing idea versus zero-padding: for every block a group does NOT have, we
    substitute a learnable embedding B_k belonging to that group. Each sample therefore hands
    the fusion block a complete set of tokens, and the model learns what "absent" means for
    each group rather than being told it is a zero vector.
    """

    def __init__(self, block_dims: Sequence[int], group_blocks: Dict[int, List[int]],
                 latent_dim: int = 64, num_classes: int = 2,
                 n_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        self.block_dims = list(block_dims)
        self.group_blocks = {int(k): list(v) for k, v in group_blocks.items()}
        self.n_blocks = len(block_dims)
        self.proj = nn.ModuleList([nn.Linear(d, latent_dim) for d in block_dims])
        # one learnable stand-in per (group, block)
        self.missing = nn.Parameter(
            torch.randn(len(self.group_blocks), self.n_blocks, latent_dim) * 0.02)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True,
                                          dropout=dropout)
        self.moe = SoftMoE(latent_dim, n_experts=n_experts, expert_hidden=latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.head = nn.Linear(latent_dim, num_classes)

    def forward(self, x_blocks: List[Optional[torch.Tensor]], g: torch.Tensor):
        """x_blocks[i] is (B, block_dims[i]) or None when no sample in the batch has it."""
        B = g.shape[0]
        toks = []
        for bi in range(self.n_blocks):
            emb = self.missing[g, bi]                       # (B, D) per-group stand-in
            if x_blocks[bi] is not None:
                real = self.proj[bi](x_blocks[bi])          # (B, D)
                has = torch.tensor(
                    [bi in self.group_blocks.get(int(gi), []) for gi in g.tolist()],
                    device=g.device).unsqueeze(-1)
                emb = torch.where(has, real, emb)
            toks.append(emb)
        z = torch.stack(toks, dim=1)                        # (B, T, D)
        a, _ = self.attn(z, z, z, need_weights=False)
        z = self.norm(z + a)
        h = self.moe(z)
        return self.head(h), h
