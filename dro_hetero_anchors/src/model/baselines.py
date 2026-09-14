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

PROVENANCE, read before quoting any of these numbers.

Flex-MoE HAS released code: github.com/UNITES-Lab/Flex-MoE, NeurIPS 2024. Ours follows its
architecture (sparse top-k gating, missing-modality bank, generalised and specialised routers)
but is NOT a port of that code, and reading their moe_module.py shows three things we do not
implement:

  - noisy gating, where noise is added to the router logits during training for exploration
  - a supervised routing loss, cross-entropy on the gate against ground-truth expert indices
  - a load-balancing loss across experts

Our specialised router hard-assigns each group to its expert; theirs supervises the gate toward
that expert while still letting it learn. Expect our Flex-MoE to be a reasonable approximation
rather than a faithful reproduction, and say so when reporting it.

REMIND has NO released code. The paper carries one URL, its own arXiv link; the abstract page
lists no repository; the corresponding author's page gives it a PDF while other projects there
carry Code links; and the appendix states code would be released after the decision, which as
of this writing has not happened. Ours is a reimplementation from the paper's description. The
Soft MoE configuration (expert count, token length) is specified only partially, so this is the
likeliest place to diverge. Report it as a reimplementation, and treat it as the
lowest-confidence number in the comparison.

Reweigh is not a separate architecture. The paper states: "we combine multi-modal MoE with
group robustness strategies", listing it alongside GroupDRO, FairBatch and FairMixup. Ours is
therefore the Soft MoE backbone plus fixed inverse-frequency group weighting, which matches
that description.

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


# ── EMBED variants ───────────────────────────────────────────────────────────────────────

class FlexMoEEmbed(nn.Module):
    """FlexMoE for EMBED: the paper's native setting, where a modality really is an image view.

    Same idea as the tabular version, but the four views (C-View CC, C-View MLO, FFDM CC,
    FFDM MLO) are the modalities and a group is the subset of views a breast actually has.
    Absent views get the group's learnable stand-in embedding B_k instead of zeros, so every
    sample presents four tokens to the Soft MoE regardless of what was scanned.

    Parameter count is reported alongside our own model because the spec asks for it on every
    row, row 4 being a different architecture.
    """

    def __init__(self, views: Sequence[str], group_views: Dict[str, Sequence[str]],
                 in_dim: int = 768, proj_dim: int = 64, latent_dim: int = 64,
                 num_classes: int = 4, n_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        self.views = list(views)
        self.groups = list(group_views.keys())
        self.group_views = {g: list(v) for g, v in group_views.items()}
        self.proj = nn.ModuleDict({v: nn.Linear(in_dim, proj_dim) for v in self.views})
        self.missing = nn.Parameter(
            torch.randn(len(self.groups), len(self.views), proj_dim) * 0.02)
        self.attn = nn.MultiheadAttention(proj_dim, num_heads=4, batch_first=True,
                                          dropout=dropout)
        self.norm = nn.LayerNorm(proj_dim)
        self.moe = SoftMoE(proj_dim, n_experts=n_experts, expert_hidden=latent_dim)
        self.head = nn.Linear(proj_dim, num_classes)

    def forward(self, group: str, view_feats: Dict[str, torch.Tensor]):
        gi = self.groups.index(group)
        any_feat = next(iter(view_feats.values()))
        B = any_feat.shape[0]
        toks = []
        for vi, v in enumerate(self.views):
            if v in view_feats:
                toks.append(self.proj[v](view_feats[v]))
            else:
                toks.append(self.missing[gi, vi].unsqueeze(0).expand(B, -1))
        z = torch.stack(toks, dim=1)                     # (B, 4, proj_dim)
        a, _ = self.attn(z, z, z, need_weights=False)
        z = self.norm(z + a)
        h = self.moe(z)
        return self.head(h), h


# ── Flex-MoE, faithful to the released implementation ────────────────────────────────────
# The first version here used a Soft MoE with a single router, which is REMIND's fusion block,
# not Flex-MoE's. The actual Flex-MoE (Yun et al., NeurIPS 2024 Spotlight,
# github.com/UNITES-Lab/flex-moe) is a SPARSE top-k MoE with two distinct routers and a
# two-phase training schedule:
#
#   missing modality bank   a learnable embedding per modality, substituted for absent ones
#   G-Router (generalised)  used during warm-up on FULL-modality samples only, so the experts
#                           first absorb knowledge that generalises across combinations
#   S-Router (specialised)  after warm-up, routes each sample with a top-1 gate to the expert
#                           owned by its observed modality combination
#
# Released defaults: 16 experts, top-k 4, hidden 128, 5 warm-up epochs.

class FlexMoESparse(nn.Module):
    """Sparse top-k MoE with a missing-modality bank and generalised/specialised routers."""

    def __init__(self, n_modalities: int, group_mods: Dict[int, List[int]],
                 in_dims: Sequence[int], d_model: int = 128, n_experts: int = 16,
                 top_k: int = 4, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_mod, self.d = n_modalities, d_model
        self.group_mods = {int(k): list(v) for k, v in group_mods.items()}
        self.n_experts, self.top_k = n_experts, top_k
        self.proj = nn.ModuleList([nn.Linear(d, d_model) for d in in_dims])
        # missing modality bank: one learnable vector per modality
        self.bank = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(d_model, d_model))
            for _ in range(n_experts)])
        self.g_router = nn.Linear(d_model, n_experts)      # generalised, used in warm-up
        self.s_router = nn.Linear(d_model, n_experts)      # specialised, top-1 after warm-up
        # each group owns one expert; groups beyond n_experts wrap around
        self.group_expert = {g: i % n_experts for i, g in enumerate(sorted(self.group_mods))}
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x_mods: List[Optional[torch.Tensor]], g: torch.Tensor,
                warmup: bool = False):
        B = g.shape[0]
        toks = []
        for mi in range(self.n_mod):
            emb = self.bank[mi].unsqueeze(0).expand(B, -1)
            if x_mods[mi] is not None:
                real = self.proj[mi](x_mods[mi])
                has = torch.tensor([mi in self.group_mods.get(int(gi), []) for gi in g.tolist()],
                                   device=g.device).unsqueeze(-1)
                emb = torch.where(has, real, emb)
            toks.append(emb)
        z = self.norm(torch.stack(toks, 1).mean(1))            # (B, d)

        if warmup:
            # G-Router: dense top-k over experts, trained on full-modality samples
            logits = self.g_router(z)
            topv, topi = logits.topk(self.top_k, dim=-1)
            w = topv.softmax(-1)
            out = z.new_zeros(B, self.d)
            for k in range(self.top_k):
                for e in range(self.n_experts):
                    m = (topi[:, k] == e)
                    if bool(m.any()):
                        out[m] += w[m, k].unsqueeze(-1) * self.experts[e](z[m])
        else:
            # S-Router: top-1 gate to the expert owned by this sample's group
            tgt = torch.tensor([self.group_expert.get(int(gi), 0) for gi in g.tolist()],
                               device=g.device)
            gate = self.s_router(z).softmax(-1)
            out = z.new_zeros(B, self.d)
            for e in range(self.n_experts):
                m = (tgt == e)
                if bool(m.any()):
                    out[m] = gate[m, e].unsqueeze(-1) * self.experts[e](z[m])
        return self.head(out), out


class FlexMoESparseEmbed(nn.Module):
    """Faithful Flex-MoE for EMBED: sparse top-k experts, missing-view bank, two routers."""

    def __init__(self, views: Sequence[str], group_views: Dict[str, Sequence[str]],
                 in_dim: int = 768, d_model: int = 128, n_experts: int = 16, top_k: int = 4,
                 num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.views = list(views)
        self.groups = list(group_views.keys())
        self.group_views = {g: list(v) for g, v in group_views.items()}
        self.n_experts, self.top_k, self.d = n_experts, top_k, d_model
        self.proj = nn.ModuleDict({v: nn.Linear(in_dim, d_model) for v in self.views})
        self.bank = nn.Parameter(torch.randn(len(self.views), d_model) * 0.02)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(d_model, d_model)) for _ in range(n_experts)])
        self.g_router = nn.Linear(d_model, n_experts)
        self.s_router = nn.Linear(d_model, n_experts)
        self.group_expert = {g: i % n_experts for i, g in enumerate(self.groups)}
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, group: str, view_feats: Dict[str, torch.Tensor], warmup: bool = False):
        B = next(iter(view_feats.values())).shape[0]
        toks = [self.proj[v](view_feats[v]) if v in view_feats
                else self.bank[i].unsqueeze(0).expand(B, -1)
                for i, v in enumerate(self.views)]
        z = self.norm(torch.stack(toks, 1).mean(1))
        out = z.new_zeros(B, self.d)
        if warmup:
            logits = self.g_router(z)
            topv, topi = logits.topk(self.top_k, dim=-1)
            w = topv.softmax(-1)
            for k in range(self.top_k):
                for e in range(self.n_experts):
                    m = (topi[:, k] == e)
                    if bool(m.any()):
                        out[m] += w[m, k].unsqueeze(-1) * self.experts[e](z[m])
        else:
            e = self.group_expert[group]
            gate = self.s_router(z).softmax(-1)[:, e].unsqueeze(-1)
            out = gate * self.experts[e](z)
        return self.head(out), out
