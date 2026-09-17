# Fed-Heart: 10-Seed Results (New Seeds)

**Seeds (fixed a priori):** 7, 13, 27, 51, 99, 137, 256, 412, 666, 1024  
**Config:** Best paper config (Adam, imbalance cap 5 for G2/G3). Only difference: baseline has `groupdro_enabled: false`, GroupDRO has `groupdro_enabled: true` plus GroupDRO hyperparameters below.

==========================================================================================
FED-HEART: 10-SEED RESULTS (NEW SEEDS)
Seeds: 7, 13, 27, 51, 99, 137, 256, 412, 666, 1024
==========================================================================================

--- SETTINGS (IDENTICAL EXCEPT GROUPDRO) ---

Shared:
  use_fedheart: true, num_classes: 2, stratified_batching: true
  group_max_train_samples: [null, null, 5, 5]  # G0,G1 full; G2,G3 capped at 5
  batch_size: 64, epochs: 100, early_stopping_patience: 30
  lr: 0.001, weight_decay: 0.0001, optimizer: adam
  grad_clip: 1.0, lr_scheduler: cosine, lr_min: 1e-5

Baseline (ERM):  groupdro_enabled: false

GroupDRO:
  groupdro_enabled: true
  groupdro_eta: 1.0, groupdro_gamma: 0.9
  groupdro_update_mode: softmax, groupdro_objective: weighted
  groupdro_kl_lambda: 0.0, groupdro_uniform_init: true

--- MODEL ARCHITECTURE ---

  Per-group encoder: MLP with LayerNorm (mlp_tabular_ln)
    input_dim: 13, hidden_dim: 64, dropout: 0.1 (×4 groups)
  Shared head: latent_dim: 64, head_hidden: 32
  Anchor/sep: anchor_eps: 0.0001, sep_samples_per_class: 4, sep_method: classifier
  lambda_fit: 0.001, lambda_sep: 0.001, sep_margin: 1.0

--- WORST-GROUP ACCURACY (mean ± std over 10 seeds) ---

  Baseline:  68.03% ± 12.99%
  GroupDRO: 68.72% ± 13.82%
  Δ:        +0.68%

--- PER-GROUP ACCURACY (mean ± std) ---

  Group              | Baseline               | GroupDRO               | Δ (pp)  
  ----------------------------------------------------------------------------
  G0 Cleveland       | 73.56% ± 6.53%   | 75.48% ± 2.65%   | +1.92%
  G1 Hungarian       | 73.15% ± 8.20%   | 77.30% ± 2.69%   | +4.16%
  G2 Swiss           | 93.75% ± 0.00%   | 93.75% ± 0.00%   | +0.00%
  G3 VA              | 68.67% ± 13.47%   | 70.44% ± 14.85%   | +1.78%

--- PER-GROUP LOSS (mean ± std) ---

  Group              | Baseline               | GroupDRO               | Δ       
  ----------------------------------------------------------------------------
  G0 Cleveland       | 0.615 ± 0.107   | 0.573 ± 0.051   | -0.042
  G1 Hungarian       | 0.563 ± 0.114   | 0.518 ± 0.058   | -0.045
  G2 Swiss           | 0.373 ± 0.135   | 0.313 ± 0.081   | -0.060
  G3 VA              | 0.887 ± 0.637   | 0.962 ± 0.617   | +0.075

==========================================================================================