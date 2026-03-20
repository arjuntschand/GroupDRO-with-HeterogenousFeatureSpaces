# Fed-Heart Disease: GroupDRO vs ERM Results

## Experimental Setup

**Dataset**: Fed-Heart Disease (UCI Heart Disease + FLamby preprocessing)
- 4 groups (hospitals): Cleveland, Hungarian, Switzerland, VA
- 13 tabular features, binary classification
- **Imbalanced setup**: G0=199, G1=172, G2=5, G3=5 training samples

**Architecture**: MLP with LayerNorm (handles small batch sizes)
- Encoder: 13 → 64 → 64 → 64 (latent dim)
- Classifier head: 64 → 32 → 2

## Results (10 Random Seeds)

| Metric | ERM Baseline | GroupDRO (Ours) | Improvement |
|--------|--------------|-----------------|-------------|
| **Worst-Group Accuracy** | 72.79% ± 7.45% | **75.31% ± 5.44%** | **+2.52%** |
| Balanced Accuracy | 80.70% ± 1.93% | 81.64% ± 1.47% | +0.93% |
| Overall Accuracy | 77.72% ± 1.35% | 78.39% ± 2.06% | +0.67% |

### Per-Group Accuracy

| Group | ERM Baseline | GroupDRO (Ours) | Improvement |
|-------|--------------|-----------------|-------------|
| G0 (Cleveland) | 76.83% ± 1.60% | 76.73% ± 1.74% | -0.10% |
| G1 (Hungarian) | 77.98% ± 1.69% | 78.88% ± 1.48% | +0.90% |
| G2 (Switzerland) | 93.75% ± 0.00% | 93.12% ± 1.98% | -0.62% |
| **G3 (VA)** | 73.56% ± 7.94% | **76.00% ± 5.72%** | **+2.44%** |

### Statistical Comparison

- **Win/Tie/Loss**: 7/3/0 (GroupDRO wins on 70% of seeds)
- **Variance Reduction**: GroupDRO has 27% lower standard deviation on worst-group accuracy (5.44% vs 7.45%)
- GroupDRO **never performs worse** than baseline across 10 seeds

## Key Findings

1. **GroupDRO consistently improves worst-group performance** with +2.52% average improvement
2. **Lower variance**: GroupDRO is more reliable across random initializations  
3. **No accuracy sacrifice**: High-performing groups (G2) maintain 93%+ accuracy
4. **Hardest group benefits most**: G3 (smallest, hardest) sees +2.44% improvement

## Optimal Configuration

```yaml
# GroupDRO Settings
groupdro_enabled: true
groupdro_eta: 1.0              # softmax temperature
groupdro_update_mode: softmax  # direct softmax (not MWU)
groupdro_uniform_init: true    # crucial for imbalanced data
groupdro_kl_lambda: 0.0        # no regularization
stratified_batching: true      # ensures minority groups appear in batches
```

## Critical Implementation Details

1. **Use LayerNorm instead of BatchNorm** for extreme imbalance (handles batch size 1)
2. **Uniform initialization** for group weights (not proportional to training sizes)
3. **Softmax update mode** is more stable than multiplicative weight update (MWU)
4. **Stratified batching is essential** - without it, GroupDRO cannot learn from minority groups

## Reproduction

```bash
# ERM Baseline
python -m dro_hetero_anchors.src.train_fedheart --config experiments/fedheart_paper_baseline.yaml

# GroupDRO
python -m dro_hetero_anchors.src.train_fedheart --config experiments/fedheart_paper_groupdro.yaml
```

## Files

- `experiments/fedheart_paper_baseline.yaml` - ERM baseline config
- `experiments/fedheart_paper_groupdro.yaml` - GroupDRO optimal config
- `dro_hetero_anchors/src/encoders/tabular_encoder.py` - MLPTabularEncoderLN (LayerNorm version)
