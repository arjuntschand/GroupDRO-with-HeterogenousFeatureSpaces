Mapping to the paper
=====================

This repo implements the main components described in the provided PDF ("GroupDRO with Heterogeneous Feature Spaces"):

- Heterogeneous encoders: per-group encoder classes are constructed from `ENCODER_REGISTRY` and trained jointly; dataset transforms and the collate function (`src/datasets.py`) allow different input sizes.
- Class anchors: implemented in `src/model/anchors.py` as parameters `m` and `L` with covariance `S = L L^T + eps I` and a normalized copy of `L` used at forward/sampling time.
- Anchor fit loss: `src/model/losses.py` computes per-class batch moments and measures Gaussian W2 (`src/model/wasserstein.py`) between batch and anchor moments.
- Anchor separation: two options are supported:
  - `sep_method: "classifier"` (default) — draw synthetic samples from anchors and train the head on them (cross-entropy).
  - `sep_method: "w2_margin"` — geometric margin on pairwise Gaussian-W2 distances between anchors (hinge loss).
  Configure in the YAML with `sep_method` and `sep_margin`.
- GroupDRO: multiplicative weights are implemented in `src/model/groupdro.py`. Weight updates are deferred until after `loss.backward()` / `optimizer.step()` to avoid in-place autograd errors; this preserves the intended multiplicative update semantics per mini-batch.

Notes / differences
-------------------
- The trainer defaults to the classifier-based separation loss for practical optimization; `w2_margin` is provided for closer fidelity to geometric objectives in the paper.
- Numerical stabilizations (adaptive regularization, Cholesky/eig fallback) are applied in `src/model/wasserstein.py` to avoid NaNs and eigendecomposition failures on nearly-singular covariances.

If you need a one-line pointer on where the PDF's equations live in code, ask and I will point to exact functions.
