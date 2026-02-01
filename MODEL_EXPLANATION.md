# Complete Model Explanation: GroupDRO with Heterogeneous Feature Spaces

## 🎯 **What This Model Does**

This model learns to classify images from **TextCaps** (a multi-modal dataset) using **two different modalities**:
- **Group 0 (Visual)**: RGB images → ResNet encoder → latent features
- **Group 1 (Text)**: Caption text → MLP encoder → latent features

The key innovation is that **both modalities map to the same latent space** and share:
1. A **shared classifier** (head)
2. **Class-wise Gaussian anchors** that represent ideal distributions for each class

**GroupDRO** ensures the model doesn't ignore the harder modality (typically text) by dynamically reweighting training to focus on groups with higher loss.

---

## 🏗️ **Architecture Overview**

```
Input (Batch)
├── Visual Samples (Group 0)
│   └── Images (B_v, 3, 224, 224)
│       └── ResNetVisualEncoder
│           └── z_v (B_v, latent_dim=64)
│
└── Text Samples (Group 1)
    └── Tokenized Captions (B_t, max_len=128)
        └── MLPTextEncoder
            └── z_t (B_t, latent_dim=64)
                │
                └── Concatenate: z = [z_v; z_t] (B, 64)
                    │
                    ├── Shared Head (MLP: 64 → 128 → 10)
                    │   └── logits (B, num_classes=10)
                    │
                    └── Anchor Module
                        └── Per-class Gaussians: N(m_c, S_c)
                            where S_c = L_c L_c^T + eps*I
```

---

## 📐 **Mathematical Formulation**

### **1. Forward Pass**

For each sample `(x, y, g)` where:
- `x`: input (image or text)
- `y`: class label ∈ {0, ..., C-1}
- `g`: group ID ∈ {0, 1}

**Step 1: Encode to latent space**
```
z = Encoder_g(x)  ∈ R^d  where d = latent_dim = 64
```

**Step 2: Classify**
```
logits = Head(z)  ∈ R^C  where C = num_classes = 10
```

**Step 3: Compute loss**
- **Baseline**: `L_CE = CrossEntropy(logits, y)`
- **GroupDRO**: `L_CE = Σ_g q_g * L_g` where `L_g` is per-group CE loss

### **2. Anchor Module**

For each class `c`, maintain a Gaussian anchor:
- **Mean**: `m_c ∈ R^d` (learnable parameter)
- **Covariance**: `S_c = L_c L_c^T + eps*I` where `L_c` is learnable

**Purpose**: Represent the "ideal" distribution for class `c` in latent space.

### **3. Loss Functions**

**Total Loss**:
```
L_total = L_CE + λ_fit * L_fit + λ_sep * L_sep
```

#### **A. Classification Loss (L_CE)**
- **Baseline**: Standard cross-entropy
- **GroupDRO**: Weighted by group weights `q_g`

#### **B. Anchor Fit Loss (L_fit)**
Minimizes 2-Wasserstein distance between **batch moments** and **anchor moments**:

```
L_fit = (1/C) * Σ_c W₂(N(μ̂_c, Σ̂_c), N(m_c, S_c))
```

Where:
- `μ̂_c, Σ̂_c`: Empirical mean/covariance of class `c` samples in current batch
- `m_c, S_c`: Anchor parameters for class `c`
- `W₂`: 2-Wasserstein distance (Bures metric)

**Purpose**: Pull representations toward class anchors.

#### **C. Anchor Separation Loss (L_sep)**
Encourages anchors to be well-separated. Two methods:

**Method 1: Classifier-based (current)**
```
For each class c:
  Sample J points from N(m_c, S_c)
  Compute: L_sep_c = CrossEntropy(Head(samples), c)
L_sep = (1/C) * Σ_c L_sep_c
```

**Method 2: W2-margin (alternative)**
```
L_sep = (1/|pairs|) * Σ_{i<j} max(0, margin - W₂(N(m_i, S_i), N(m_j, S_j)))
```

**Purpose**: Ensure anchors don't collapse to the same point.

### **4. GroupDRO Weight Update**

**Multiplicative Weights Update (MWU)**:
```
q_g^(t+1) = (q_g^(t) * exp(η * L_g)) / (Σ_h q_h^(t) * exp(η * L_h))
```

With **exponential smoothing**:
```
q_new = (1 - γ) * q_old + γ * q_MWU
```

Where:
- `η` (eta): Learning rate for weight updates
- `γ` (gamma): Smoothing factor (0.9 = strong smoothing)

**Purpose**: Automatically upweight groups with higher loss.

---

## ⚙️ **Complete Hyperparameter Reference**

### **Dataset Configuration**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `textcaps_use_huggingface` | `true` | Load from HF (recommended) vs local JSON |
| `textcaps_num_classes` | `10` | Top-K most frequent classes to use |
| `textcaps_max_train_samples` | `None` | Limit training samples (None = use all ~22k) |
| `textcaps_max_test_samples` | `None` | Limit test samples (None = use all ~3k) |
| `batch_size` | `32` | Samples per batch |
| `num_workers` | `0` | DataLoader workers (0 = single-threaded) |

### **Model Architecture**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `latent_dim` | `64` | Dimension of shared latent space |
| `num_classes` | `10` | Number of classification classes |
| `head_hidden` | `128` | Hidden dim for MLP head (0 = linear head) |
| `groups[0].encoder` | `resnet_visual` | Visual encoder type |
| `groups[1].encoder` | `mlp_text` | Text encoder type |

**Available Encoders**:
- **Visual**: `resnet_visual` (pretrained ResNet18), `simple_cnn_visual`
- **Text**: `mlp_text`, `char_cnn_text`, `transformer_text`

### **Anchor Module**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `anchor_eps` | `1.0e-4` | Regularization for covariance: `S = L L^T + eps*I` |
| `sep_samples_per_class` | `16` | Number of samples `J` per class for separation loss |
| `sep_method` | `"classifier"` | `"classifier"` or `"w2_margin"` |
| `sep_margin` | `2.0` | Margin for W2-margin method (if used) |

### **Loss Weights**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lambda_fit` | `1.0e-3` | Weight for anchor fit loss |
| `lambda_sep` | `1.0e-3` | Weight for anchor separation loss |

**Trade-offs**:
- Higher `lambda_fit` → stronger pull toward anchors (may reduce flexibility)
- Higher `lambda_sep` → stronger separation (may improve class boundaries)

### **Optimizer**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `optimizer` | `adam` | Optimizer type (only Adam supported) |
| `lr` | `1.0e-3` | Learning rate |
| `weight_decay` | `1.0e-4` | L2 regularization |
| `grad_clip` | `1.0` | Gradient clipping norm (0 = disabled) |
| `epochs` | `10` | Number of training epochs |

### **GroupDRO (when enabled)**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `groupdro_enabled` | `true/false` | Enable GroupDRO |
| `groupdro_eta` | `0.01` | Learning rate for weight updates (lower = slower adaptation) |
| `groupdro_gamma` | `0.9` | Smoothing factor (0.9 = strong smoothing, 1.0 = no smoothing) |
| `groupdro_update_mode` | `"exp_smooth"` | `"exp"`, `"softmax"`, or `"exp_smooth"` |
| `groupdro_objective` | `"weighted"` | `"weighted"`, `"max"`, or `"logsumexp"` |
| `groupdro_warmup_epochs` | `2` | Epochs with uniform weights before GroupDRO kicks in |

**GroupDRO Modes Explained**:

1. **`update_mode: "exp"`**: Pure MWU (can be unstable)
2. **`update_mode: "softmax"`**: Direct softmax projection (no momentum)
3. **`update_mode: "exp_smooth"`**: MWU + exponential smoothing (recommended)

**Objective Modes**:
- **`"weighted"`**: `Σ_g q_g * L_g` (standard DRO surrogate)
- **`"max"`**: `max_g L_g` (worst-group ERM)
- **`"logsumexp"`**: Smooth approximation to max

### **Logging & Checkpointing**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `log_interval` | `20` | Log to TensorBoard every N batches |
| `save_every` | `5` | Save checkpoint every N epochs |
| `run_dir` | `runs/...` | Output directory for logs/checkpoints |

---

## 🔄 **Training Loop (Step-by-Step)**

### **Per Batch:**

1. **Load batch**: Mixed visual + text samples
   ```
   visual_x: (B_v, 3, 224, 224)
   text_x: (B_t, 128)  # token IDs
   ```

2. **Encode**:
   ```
   z_v = Encoder_0(visual_x)  # (B_v, 64)
   z_t = Encoder_1(text_x)    # (B_t, 64)
   z = concat([z_v, z_t])      # (B, 64)
   ```

3. **Classify**:
   ```
   logits = Head(z)  # (B, 10)
   ```

4. **Compute losses**:
   ```
   L_CE = GroupDRO.forward(logits, y, g)  # or standard CE
   
   # Anchor fit: compare batch moments to anchors
   moments = per_class_batch_moments(z, y, num_classes, eps)
   m_anc, S_anc, L_norm = anchors.forward()
   L_fit = anchor_fit_loss(m_anc, S_anc, moments, eps)
   
   # Anchor separation: sample from anchors, classify
   L_sep = anchor_sep_loss(m_anc, S_anc, L_norm, head, ...)
   
   L_total = L_CE + λ_fit * L_fit + λ_sep * L_sep
   ```

5. **Backward & update**:
   ```
   L_total.backward()
   clip_grad_norm(params, grad_clip)
   optimizer.step()
   ```

6. **Update GroupDRO weights** (if enabled):
   ```
   if epoch > warmup_epochs:
       groupdro.update_weights(group_losses, group_counts)
   ```

### **Per Epoch:**

1. Train on all batches
2. Evaluate on test set (separate by group)
3. Log metrics:
   - Overall accuracy
   - Per-group accuracy (visual, text)
   - Worst-group accuracy (min of group accuracies)
   - Balanced accuracy (mean of group accuracies)
   - GroupDRO weights `q`

---

## 🎛️ **Hyperparameter Tuning Guidelines**

### **For Better Accuracy**:
- Increase `epochs` (10 → 50+)
- Increase `latent_dim` (64 → 128) if overfitting isn't an issue
- Use `resnet_visual` (pretrained) instead of `simple_cnn_visual`
- Increase `head_hidden` (128 → 256) for more capacity

### **For Better Group Balance** (GroupDRO):
- **Lower `groupdro_eta`** (0.01 → 0.001): Slower weight adaptation, more stable
- **Higher `groupdro_gamma`** (0.9 → 0.95): More smoothing, less oscillation
- **Increase `groupdro_warmup_epochs`** (2 → 5): Let model learn before reweighting

### **For Faster Training**:
- Reduce `batch_size` (32 → 16) if OOM
- Set `num_workers` > 0 for data loading
- Use `simple_cnn_visual` instead of `resnet_visual`
- Reduce `sep_samples_per_class` (16 → 8)

### **For Better Anchor Alignment**:
- Increase `lambda_fit` (1e-3 → 1e-2): Stronger pull to anchors
- Increase `lambda_sep` (1e-3 → 1e-2): Stronger separation
- Adjust `anchor_eps` (1e-4 → 1e-5): Tighter covariance regularization

---

## 📊 **Key Metrics Tracked**

1. **Overall Accuracy**: `correct / total` across all groups
2. **Per-Group Accuracy**: Separate for visual and text
3. **Worst-Group Accuracy**: `min(acc_visual, acc_text)` ← **Primary metric for GroupDRO**
4. **Balanced Accuracy**: `(acc_visual + acc_text) / 2`
5. **GroupDRO Weights**: `q[0]` (visual), `q[1]` (text) - should be balanced if working well

---

## 🔍 **What to Watch For**

### **Good Signs**:
- ✅ Worst-group accuracy increasing
- ✅ GroupDRO weights `q` balanced (not collapsing to one group)
- ✅ Both visual and text accuracies improving
- ✅ Anchor fit loss decreasing (anchors aligning with data)

### **Warning Signs**:
- ⚠️ GroupDRO weights collapse (`q[0] ≈ 1.0` or `q[1] ≈ 1.0`)
  - **Fix**: Lower `groupdro_eta`, increase `groupdro_gamma`
- ⚠️ One group accuracy much lower than the other
  - **Fix**: Increase `groupdro_warmup_epochs`, check encoder capacity
- ⚠️ Anchor fit loss not decreasing
  - **Fix**: Increase `lambda_fit`, check `anchor_eps`
- ⚠️ Training loss not decreasing
  - **Fix**: Lower learning rate, check data loading

---

## 🚀 **Next Steps for Experimentation**

1. **Run baseline** (no GroupDRO) to establish baseline accuracies
2. **Run GroupDRO** and compare worst-group accuracy
3. **Tune GroupDRO hyperparameters** if weights collapse
4. **Increase epochs** (10 → 50+) for better convergence
5. **Experiment with encoders**: Try `transformer_text` for better text encoding
6. **Try different separation methods**: Switch to `w2_margin` if classifier method fails

---

This model is designed to learn robust representations that work across heterogeneous modalities (images vs text) by aligning them to a shared latent space with class anchors, while GroupDRO ensures balanced learning across groups.
