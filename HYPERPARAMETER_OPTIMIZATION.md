# Hyperparameter Optimization Analysis & Recommendations

## 📊 Current Configuration Analysis

### Current Settings (TextCaps)
| Parameter | Current Value | Status |
|-----------|--------------|--------|
| **Optimizer** | Adam | ✅ Good |
| **Learning Rate** | 1e-3 | ⚠️ May be too high for pretrained ResNet |
| **Weight Decay** | 1e-4 | ✅ Standard |
| **Batch Size** | 32 | ⚠️ Could be larger (64-128) |
| **Latent Dim** | 64 | ⚠️ Small for multi-modal (consider 128-256) |
| **Head Hidden** | 128 | ✅ Reasonable |
| **Classes** | 10 | ✅ Good for initial experiments |
| **Epochs** | 10 | ⚠️ Too few (need 50+ for convergence) |
| **Text Encoder** | MLP | ⚠️ Simple (CharCNN/Transformer may be better) |
| **LR Scheduler** | None | ❌ Missing (should add) |
| **Gradient Clip** | 1.0 | ✅ Good |

---

## 🎯 Recommended Optimizations

### 1. **Learning Rate Strategy** ⭐ HIGH PRIORITY

**Problem**: Using `lr=1e-3` for all parameters, but ResNet backbone is pretrained and should be fine-tuned with lower LR.

**Recommendation**: **Differential Learning Rates**

```yaml
# Option A: Separate LRs for pretrained vs new layers
lr_backbone: 1.0e-4  # Lower for pretrained ResNet
lr_new: 1.0e-3       # Higher for new layers (head, anchors, text encoder)
```

**OR**

```yaml
# Option B: Use LR scheduler (linear decay)
lr: 1.0e-3
lr_start: 1.0e-3
lr_end: 1.0e-5
# Decay linearly over epochs
```

**OR**

```yaml
# Option C: StepLR scheduler (reduce on plateau)
lr: 1.0e-3
lr_scheduler: "StepLR"
lr_step_size: 15
lr_gamma: 0.1
```

**Best Practice**: Use **Option A** (differential LRs) for best results with pretrained models.

---

### 2. **Batch Size** ⭐ MEDIUM PRIORITY

**Current**: `batch_size: 32`

**Recommendation**: 
- **64** for MPS/GPU (better gradient estimates, faster training)
- **128** if you have enough memory
- Keep **32** if memory-constrained

**Trade-offs**:
- Larger batch → more stable gradients, faster training, but may need higher LR
- Smaller batch → more gradient noise (can help generalization), slower

**Suggested**: `batch_size: 64`

---

### 3. **Latent Dimension** ⭐ MEDIUM PRIORITY

**Current**: `latent_dim: 64`

**Problem**: ResNet outputs 512-dim features, but we're compressing to 64. This may lose information.

**Recommendation**:
- **128** (good balance, 2x current)
- **256** (more capacity, better for complex multi-modal alignment)
- Keep **64** if you want faster training and smaller model

**Suggested**: `latent_dim: 128` or `latent_dim: 256`

**Note**: If you increase latent_dim, also consider increasing `head_hidden` proportionally.

---

### 4. **Text Encoder** ⭐ HIGH PRIORITY

**Current**: `mlp_text` (simple bag-of-characters)

**Problem**: MLP doesn't capture sequential structure of text well.

**Recommendation**:
- **`char_cnn_text`**: Better for OCR text, captures local patterns
- **`transformer_text`**: Best for complex text, but slower

**Suggested**: Try `char_cnn_text` first (good balance of performance/speed)

---

### 5. **Head Architecture** ⭐ LOW PRIORITY

**Current**: `head_hidden: 128` (MLP: 64 → 128 → 10)

**Recommendation**:
- If `latent_dim` increases to 128: `head_hidden: 256`
- If `latent_dim` increases to 256: `head_hidden: 512`
- Keep current if staying at `latent_dim: 64`

**Suggested**: Scale proportionally with `latent_dim`

---

### 6. **Number of Classes** ⭐ LOW PRIORITY

**Current**: `textcaps_num_classes: 10`

**Options**:
- **10**: Good for initial experiments, easier to learn
- **15-20**: More realistic, better for paper results
- **All classes**: Most challenging, may need more data/epochs

**Recommendation**: Start with **10**, then try **15-20** after baseline works.

---

### 7. **Learning Rate Scheduler** ⭐ HIGH PRIORITY

**Current**: None (constant LR)

**Recommendation**: Add one of:

**Option A: Linear Decay** (simple, works well)
```yaml
lr: 1.0e-3
lr_start: 1.0e-3
lr_end: 1.0e-5
```

**Option B: StepLR** (reduce every N epochs)
```yaml
lr: 1.0e-3
lr_scheduler: "StepLR"
lr_step_size: 15
lr_gamma: 0.1
```

**Option C: Cosine Annealing** (smooth decay)
```yaml
lr: 1.0e-3
lr_scheduler: "CosineAnnealingLR"
T_max: 50  # epochs
eta_min: 1.0e-5
```

**Suggested**: **Option A** (linear decay) - easiest to implement and works well.

---

### 8. **Optimizer** ⭐ LOW PRIORITY

**Current**: Adam (`lr=1e-3`)

**Status**: ✅ **Good choice** for multi-modal learning

**Alternatives** (if Adam doesn't work well):
- **AdamW**: Better weight decay handling (use `weight_decay: 1e-4`)
- **SGD with momentum**: Used in some MNIST/USPS experiments, but Adam is generally better for vision

**Recommendation**: **Keep Adam**, but consider **AdamW** if you want better regularization.

---

### 9. **Epochs** ⭐ HIGH PRIORITY

**Current**: `epochs: 10` (for testing)

**Recommendation**: 
- **50-100** for real experiments
- **20-30** for quick validation
- Monitor validation loss to stop early if needed

**Suggested**: `epochs: 50` for initial full runs, then increase if needed.

---

### 10. **GroupDRO Hyperparameters** ⭐ MEDIUM PRIORITY

**Current**:
- `groupdro_eta: 0.01` ✅ Good (prevents collapse)
- `groupdro_gamma: 0.9` ✅ Good (smoothing)
- `groupdro_warmup_epochs: 2` ⚠️ Could be longer

**Recommendation**:
- Keep `eta: 0.01` (or try `0.005` if still collapsing)
- Keep `gamma: 0.9` (or try `0.95` for more smoothing)
- Increase `warmup_epochs: 5` (let model learn before reweighting)

---

### 11. **Loss Weights** ⭐ LOW PRIORITY

**Current**:
- `lambda_fit: 1e-3`
- `lambda_sep: 1e-3`

**Status**: ✅ Reasonable starting point

**Tuning**:
- If anchors not aligning: increase `lambda_fit` to `1e-2`
- If classes overlapping: increase `lambda_sep` to `1e-2`
- If overfitting: decrease both to `1e-4`

---

### 12. **Anchor Parameters** ⭐ LOW PRIORITY

**Current**:
- `anchor_eps: 1e-4` ✅ Good
- `sep_samples_per_class: 16` ✅ Reasonable
- `sep_method: "classifier"` ✅ Good

**Recommendation**: Keep as-is unless you see issues.

---

## 🚀 **Recommended Configuration (Optimized)**

### **Baseline (No GroupDRO)**
```yaml
# Optimizer
optimizer: adam
lr: 1.0e-3
lr_backbone: 1.0e-4  # For pretrained ResNet
lr_new: 1.0e-3        # For new layers
weight_decay: 1.0e-4
grad_clip: 1.0

# Training
batch_size: 64
epochs: 50
lr_start: 1.0e-3
lr_end: 1.0e-5

# Architecture
latent_dim: 128  # Increased from 64
head_hidden: 256  # Scaled with latent_dim
textcaps_num_classes: 10

# Encoders
groups:
  - encoder: resnet_visual  # Pretrained
  - encoder: char_cnn_text  # Better than MLP

# Anchors (unchanged)
anchor_eps: 1.0e-4
sep_samples_per_class: 16
sep_method: "classifier"
lambda_fit: 1.0e-3
lambda_sep: 1.0e-3
```

### **GroupDRO (Optimized)**
```yaml
# Same as baseline, plus:
groupdro_enabled: true
groupdro_eta: 0.01
groupdro_gamma: 0.9
groupdro_warmup_epochs: 5  # Increased from 2
groupdro_update_mode: exp_smooth
groupdro_objective: weighted
```

---

## 📈 **Implementation Priority**

### **Phase 1: Quick Wins** (Do First)
1. ✅ Increase `batch_size: 32 → 64`
2. ✅ Increase `epochs: 10 → 50`
3. ✅ Switch `mlp_text → char_cnn_text`
4. ✅ Add LR scheduler (linear decay)

### **Phase 2: Architecture** (After Phase 1 works)
5. ✅ Increase `latent_dim: 64 → 128`
6. ✅ Scale `head_hidden: 128 → 256`
7. ✅ Implement differential LRs (backbone vs new)

### **Phase 3: Fine-tuning** (After Phase 2)
8. ✅ Tune GroupDRO warmup epochs
9. ✅ Try `latent_dim: 256` if 128 works well
10. ✅ Experiment with more classes (15-20)

---

## 🔬 **Expected Impact**

| Change | Expected Impact | Risk |
|--------|----------------|------|
| **Batch 32→64** | +5-10% accuracy, faster training | Low |
| **Epochs 10→50** | +10-20% accuracy | Low |
| **MLP→CharCNN** | +5-15% text accuracy | Low |
| **Latent 64→128** | +3-8% overall accuracy | Medium |
| **LR scheduler** | Better convergence, +2-5% | Low |
| **Differential LR** | +5-10% accuracy (ResNet fine-tuning) | Low |

---

## ⚠️ **Notes**

1. **Differential Learning Rates**: Requires code changes to `train_textcaps.py` to set different LRs for different parameter groups.

2. **LR Scheduler**: `train_textcaps.py` currently doesn't support schedulers. Need to add support (similar to `train.py` which has linear decay).

3. **Memory**: Increasing batch size and latent_dim will use more GPU memory. Monitor MPS usage.

4. **Training Time**: More epochs + larger batches = longer training. Plan accordingly.

5. **Validation**: Always validate on held-out test set to avoid overfitting.

---

## 🎯 **Next Steps**

1. **Implement Phase 1 changes** (quick wins)
2. **Run baseline experiment** with optimized config
3. **Compare results** to current baseline
4. **Iterate** with Phase 2/3 changes if needed

---

**Bottom Line**: The current config is reasonable for initial testing, but these optimizations should significantly improve performance for your conference paper results.
