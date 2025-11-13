DRO-HeteroAnchors — Centralized Baseline
========================================

Centralized latent-anchor alignment for heterogeneous feature spaces
(inspired by FLIC, but no federated orchestration).

**Core idea**
- Per-group encoders φ_g map inputs to a shared latent ℝ^k.
- For each class c, learn a Gaussian anchor μ_c = N(m_c, S_c), S_c = L_c L_cᵀ + εI.
- Loss = classification + W₂ (Bures) alignment of batch class moments to anchors
  + separation via classifying synthetic samples drawn from anchors.

Quickstart
----------
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.train --config experiments/digits_centralized.yaml
python -m src.eval --config experiments/digits_centralized.yaml --ckpt runs/last.ckpt

