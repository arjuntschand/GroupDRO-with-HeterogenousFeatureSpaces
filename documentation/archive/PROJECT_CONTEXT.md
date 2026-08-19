# PROJECT_CONTEXT.md — Canonical Reference

> **Purpose of this file:** Single source of truth for what this project *is*, the method, the
> datasets, the current results, and where everything lives. Read this first when starting fresh
> or after losing context. Last substantively updated: 2026-07-18.

---

## 1. One-paragraph summary

Research project on **GroupDRO across heterogeneous feature spaces**. Standard ERM and GroupDRO
(Sagawa et al., ICLR 2020) assume every group shares one input/feature representation, so they fail
to protect minority groups when groups genuinely live in *different* feature spaces (different image
resolutions, different tabular columns, different modalities). We fix this with: **per-group
encoders φ_g** that map each group's native input into a **shared latent space**, **per-class
Gaussian anchors** that align groups in that space via a 2-Wasserstein penalty, and a **GroupDRO
min–max objective** that focuses on the worst group. Goal metric is **worst-group accuracy**.

**Headline result:** per-group encoders + Gaussian anchors + GroupDRO improve worst-group accuracy
by **+7.9% to +15%** over ERM baselines across three datasets (5–10 seeds).

**THE key intellectual finding:** anchors and GroupDRO are **synergistic** — neither helps much
alone, but *combined* they produce large gains, because the anchors structure the shared latent
space so that GroupDRO can effectively reweight across heterogeneous groups. This synergy, not the
raw GroupDRO delta, is the paper's main point.

## 2. People & status

- **Arjun Tschand** — author. Duke undergrad, ECE + CS double major, graduating ~mid-2027.
- **Xenia Konti** — PhD student, direct collaborator. Results are communicated to her by email
  (casual tone, no em dashes, tables preferred for copy-paste into Outlook). She designed the
  NHANES dataset direction.
- **Michael Zavlanos** — faculty advisor.
- **Status/goal (as of 2026-07-18):** finish results + writing and land a **publication or workshop
  publication within ~3 months**, before grad-school applications. Poster already presented at Duke
  Spring 2026 (`documentation/Arjun Tschand Research Poster Spring 2026.pdf`).

## 3. Method (what the model actually does)

For each sample `(x, y, g)` — input `x`, label `y ∈ {0..C-1}`, group `g`:

1. **Per-group encoder:** `z = φ_g(x) ∈ ℝ^k` (shared latent dim `k`). Each group has its **own**
   encoder; with `true_hetero_input_dim`, each encoder's input dim matches that group's actual
   feature count. A `common_encoder: true` flag makes all groups share one encoder (baseline).
2. **Shared classifier head:** `logits = ψ(z) ∈ ℝ^C`. One head for all groups.
3. **Per-class Gaussian anchors:** each class `c` has a learnable Gaussian `N(m_c, S_c)` where
   `S_c = L_c L_cᵀ + εI` (learnable mean `m_c` and low-rank factor `L_c`). Anchors model
   *covariance*, not just a prototype mean.
4. **Total loss:** `L_total = L_CE + λ_fit · L_fit + λ_sep · L_sep`
   - `L_CE` — classification cross-entropy. Standard, or **GroupDRO-weighted** `Σ_g q_g · L_g`.
   - `L_fit` — **anchor fit:** `(1/C) Σ_c W₂( N(μ̂_c, Σ̂_c), N(m_c, S_c) )`, the squared
     2-Wasserstein (Bures) distance between the batch's empirical class Gaussian and the anchor.
     Pulls every group's class-`c` embeddings to a common region.
   - `L_sep` — **anchor separation:** keeps class anchors apart / prevents collapse. Two methods:
     `classifier` (sample J points from each anchor, cross-entropy them to class `c`) or
     `w2_margin` (hinge on pairwise W₂ between anchors).
5. **GroupDRO weight update:** maintains `q` on the simplex; upweights high-loss groups. Modes:
   update `exp` / `softmax` / `exp_smooth`; objective `weighted` / `max` / `logsumexp`; optional
   `KL(q‖π)` penalty. For imbalanced tabular data, **`softmax` + `uniform_init` + stratified
   batching** is the reliable combo.

Pipeline: `YAML config → group-aware loaders → per-group φ_g → shared latent → head + anchors →
L_total → (GroupDRO reweight) → backward`.

## 4. Datasets (three in the paper + one extra)

| Dataset | Heterogeneity | Groups | Task | Trainer |
|---|---|---|---|---|
| **MNIST/USPS** | Mixed image resolution (28×28 vs 16×16) | 2 | 10-class digits | `train.py` |
| **Fed-Heart** | Different clinical features per hospital (FLamby/UCI) | 4 hospitals | Binary heart disease | `train_fedheart.py` |
| **NHANES CVD** | *Natural* feature availability (survey ⊂ exam ⊂ labs) | 3 | Binary CVD | `train_nhanes.py` |
| TextCaps | True multimodal (image vs text) | 2 | Multi-class | `train_textcaps.py` |

- **NHANES is the strongest argument** — heterogeneity is real/natural, not artificially masked.
- **TextCaps is the weak spot** and was NOT on the poster's headline results. GroupDRO there tends
  to only hurt the best group (weak text-encoder ceiling). Decide if it belongs in the paper.

### Fed-Heart feature index (after preprocessing, 13 features)
`0:age, 1:sex, 2:trestbps, 3:chol, 4:fbs, 5:thalach, 6:exang, 7:oldpeak, 8-10:cp(one-hot),
11-12:restecg(one-hot)`. Original UCI `slope`, `ca`, `thal` are **dropped** in preprocessing
(`datasets_fedheart.py`).

### NHANES feature index (nested mode, up to 20 features)
`0:age, 1:gender, 2-6:race(one-hot), 7:education, 8:income_poverty, 9:ever_smoked, 10:bmi,
11:weight, 12:height, 13:mean_systolic_bp, 14:mean_diastolic_bp, 15:hba1c, 16:hdl,
17:total_cholesterol, 18:triglycerides, 19:ldl`. Feature modes: `nested` (10/13/20, G0⊂G1⊂G2),
`expanded` (15/18/25, more survey features), `disjoint` (15/15/15, 10 shared + 5 unique per group).

## 5. Current results (what to cite)

**Fed-Heart** (10 seeds, hetero features, `data_split_seed=43`, cap `[None,None,10,15]`, GDRO
`eta=2.0` softmax): worst-group accuracy
- Shared ERM 65.14% → Shared GDRO 71.52% → Per-group ERM 74.05% → **Per-group GDRO 74.25% (+9.12%)**

**NHANES CVD** (5 seeds, `data_split_seed=100`, auto class-weighting for 10.5% CVD rate):
worst-group accuracy
- Nested: Shared ERM 68.97 → **Shared GDRO 71.37 (+2.40)**
- Expanded: Shared ERM 74.30 → **Shared GDRO 76.91 (+2.61)** ← best overall
- Disjoint: Shared ERM 70.92 → **Per-group GDRO 72.69 (+1.77)**
- AUROC ~80% and stable across configs; accuracy gaps come from calibration/threshold effects.

**Cross-dataset takeaways:**
- GroupDRO **consistently raises worst-group accuracy and cuts variance**.
- **Per-group encoders win when features are genuinely different** (Fed-Heart, NHANES-disjoint);
  **shared encoder wins/ties when features are nested/overlapping** (NHANES-nested/expanded). This
  is an honest, expected result — state it, don't hide it.

**Known caveats to defend before submission:**
1. Fed-Heart's giant +36% GroupDRO number comes from *rescuing a mis-tuned SGD-low-LR baseline*.
   The fair well-tuned-Adam gain is ~+3.86%. Don't headline the +36% naively.
2. NHANES **sensitivity/recall can drop under GDRO** even as accuracy rises (e.g. G1 recall
   67.9%→59.2%) — for a clinical task, reviewers will care; needs an explicit story.
3. TextCaps worst-group gains are fragile (weak text encoder ceiling).

## 6. Repository map

```
CLAUDE.md                     # Build/run commands + per-dataset config reference (authoritative for commands)
README.md                     # Purpose, architecture, config schema
MODEL_EXPLANATION.md          # Deep method + hyperparameter walkthrough (TextCaps-centric)
HYPERPARAMETER_OPTIMIZATION.md
documentation/
  PAPER_IMPLEMENTATION_GUIDE.md
  archive/PROJECT_CONTEXT.md  # <-- THIS FILE (canonical reference)
  Arjun Tschand Research Poster Spring 2026.pdf   # paper framing / headline claims
dro_hetero_anchors/src/
  train.py / train_fedheart.py / train_nhanes.py / train_textcaps.py   # per-dataset training loops
  datasets*.py                # per-dataset group-aware loaders
  eval.py
  encoders/                   # cnn28, cnn32, resnet_visual, tabular_encoder, text_encoder; ENCODER_REGISTRY
  model/anchors.py            # AnchorModule N(m_c, S_c)
  model/losses.py             # anchor fit/sep, per_class_batch_moments, focal, label smoothing
  model/groupdro.py           # GroupDRO class (q on simplex, update/objective modes, KL penalty)
  model/wasserstein.py        # gaussian_w2 (Bures), psd_sqrt
  model/head.py               # shared classifier
  tools/                      # index_experiments, aggregate_runs
experiments/                  # 100+ YAML configs (+ INDEX.csv)
runs/                         # run outputs + results markdown (see below)
run_experiments.py / run_nhanes_experiments.py / run_nhanes_all.py    # batch runners
```

**Results markdown lives in `runs/`** — key ones: `NHANES_EXPERIMENT_RESULTS.md`,
`FEDHEART_FINAL_RESULTS.md`, `HETERO_EXPERIMENT_RESULTS.md`,
`COMPREHENSIVE_EXPERIMENTS_SUMMARY.md`. Note: several results MDs exist with overlapping numbers —
reconcile against raw run logs before quoting.

## 7. Commands (quick)

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r dro_hetero_anchors/requirements.txt

# Train (per dataset)
python -m dro_hetero_anchors.src.train           --config experiments/<cfg>.yaml   # MNIST/USPS
python -m dro_hetero_anchors.src.train_fedheart  --config experiments/<cfg>.yaml   # Fed-Heart
python -m dro_hetero_anchors.src.train_nhanes    --config experiments/<cfg>.yaml   # NHANES
python -m dro_hetero_anchors.src.train_textcaps  --config experiments/<cfg>.yaml   # TextCaps

# Batch experiment suites
python run_nhanes_experiments.py --seeds 42 1337 7 --only core
python run_nhanes_all.py    # nested × expanded × disjoint × shared/pergroup × ERM/GDRO

# Eval / tooling / tests
python -m dro_hetero_anchors.src.eval --config <cfg.yaml> --ckpt <ckpt.pt>
python -m dro_hetero_anchors.src.tools.aggregate_runs
python -m pytest dro_hetero_anchors/tests/
```

Datasets auto-download on first run into `datasets/` (gitignored).

## 8. Key config flags to remember

- `common_encoder: true` — shared encoder baseline (default = per-group).
- `true_hetero_input_dim: true` — each per-group encoder gets its real feature count (required for
  genuine heterogeneity experiments).
- `feature_mask` — per-group list of feature indices to keep (`null` = keep all).
- `data_split_seed` — fixes the train/test partition (like FLamby); model `seed` then only controls
  init/subsampling. **Use this for stable, comparable multi-seed results.**
- `group_max_train_samples` — cap samples per group to induce imbalance.
- GroupDRO: `groupdro_enabled`, `groupdro_eta`, `groupdro_update_mode` (softmax reliable),
  `groupdro_objective` (weighted; `max` tends to destroy performance), `groupdro_uniform_init`,
  `groupdro_kl_lambda`.
- NHANES: `class_weight: "auto"`, `use_post_pandemic: true`, `feature_mode: nested|expanded|disjoint`.

## 9. Positioning / references

- **[1] Rakotomamonjy et al., "Personalised federated learning on heterogeneous feature spaces"**
  (arXiv 2301.11447, 2023) — closest prior work; the method to distinguish ourselves from.
- **[2] FLamby** (NeurIPS D&B 2022) — source of Fed-Heart.
- **[3] Sagawa et al., GroupDRO** (ICLR 2020) — the base method we extend.

**Stated future work:** more datasets, theoretical convergence guarantees, adapting to federated
learning settings.

## 10. Working preferences

Keep explanations accessible but technically precise. Use 2 seeds to screen configs, scale to 10
only on promising ones. Match original/known-good settings when comparing. Prefer speed + iteration
over perfection; surface meaningful improvements early. Emails to Xenia: casual tone, no em dashes,
results in copy-pasteable tables.
