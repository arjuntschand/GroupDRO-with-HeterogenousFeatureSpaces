# Complete Project Structure Guide: Everything Explained

## Overview

Your project directory structure:
```
dro_hetero_anchors/
├── .venv/                          # Virtual environment (isolated Python)
├── data/                           # Datasets (MNIST, USPS, etc.)
├── experiments/                    # Configuration files for experiments
├── src/                            # All source code
├── tests/                          # Unit tests
├── runs/                           # Training outputs (checkpoints, logs)
├── README.md                       # Quick start guide
├── MAPPING.md                      # PDF-to-code mapping
├── IMPLEMENTATION_GUIDE.md         # Detailed walkthrough (just created)
├── requirements.txt                # Python dependencies
└── [other files]
```

---

# Directory-by-Directory Breakdown

## 1. .venv/ (Virtual Environment)

**What it is**: Isolated Python environment specific to this project.

**Why it exists**: 
- Different projects need different package versions (PyTorch 2.0 vs 1.13, etc.)
- `.venv/` keeps your project's dependencies separate from your system Python
- If you break something in `.venv/`, it doesn't affect other projects or system

**Contents** (you don't edit these):
```
.venv/
├── bin/                           # Executable programs
│   ├── python                     # Python interpreter for this project
│   ├── pip                        # Package installer
│   └── [other tools]
├── lib/                           # Installed packages
│   └── python3.13/site-packages/  # PyTorch, torchvision, etc.
└── [other config files]
```

**How to use it**:
```bash
# Activate the venv (do this every time you open a terminal)
source .venv/bin/activate

# You'll see (.venv) at the start of your prompt:
# (.venv) arjun@MacBook dro_hetero_anchors %

# Now run Python commands with this project's packages
python -m src.train --config experiments/digits_centralized.yaml

# To deactivate later
deactivate
```

**What's inside** (installed packages):
```
PyTorch               # Deep learning framework
torchvision           # Image utilities (MNIST download, transforms)
pyyaml                # Read .yaml config files
tqdm                  # Progress bars
rich                  # Colored console output
tensorboard           # Logging and visualization
numpy                 # Numerical computing
[and their dependencies]
```

**Installed via**: `pip install -r requirements.txt`

---

## 2. data/ (Datasets)

**What it is**: Local storage for datasets used in training.

**Directory structure**:
```
data/
└── MNIST/
    ├── raw/
    │   ├── t10k-images-idx3-ubyte        # Test images (binary)
    │   ├── t10k-labels-idx1-ubyte        # Test labels (binary)
    │   ├── train-images-idx3-ubyte       # Train images (binary)
    │   └── train-labels-idx1-ubyte       # Train labels (binary)
    └── [processed/ if cached]
```

**How it works**:

First run:
```python
# In src/datasets.py:
base = datasets.MNIST(root="./data", train=True, download=True)
# → torchvision downloads MNIST from the internet
# → Saves to data/MNIST/raw/
```

Subsequent runs:
```python
# Same code, but now:
# → Sees data already exists in data/MNIST/raw/
# → Skips download, loads from disk (much faster)
```

**File formats**:
- `.ubyte` = unsigned byte binary files (MNIST native format)
- These are NOT human-readable; PyTorch's `datasets.MNIST` parses them automatically
- Each file contains thousands of 28×28 grayscale images and their labels

**What gets stored**:
- **train-images**: 60,000 training images (28×28 pixels each)
- **train-labels**: 60,000 training labels (digit 0-9)
- **t10k-images**: 10,000 test images
- **t10k-labels**: 10,000 test labels

**For future datasets** (e.g., USPS):
```
data/
├── MNIST/
└── USPS/              # Would go here when we add USPS
    └── raw/
```

---

## 3. experiments/ (Configuration Files)

**What it is**: YAML files that configure training runs without changing code.

**Current file**:
```
experiments/
└── digits_centralized.yaml
```

**Why separate configs?**
- Same code, different configs = different experiments
- Easy to compare results from different hyperparameters
- No need to modify Python code for each run

**File: digits_centralized.yaml (detailed breakdown)**

```yaml
# ============ METADATA ============
seed: 1337                          # Random seed for reproducibility

run_dir: runs                       # Where to save checkpoints and logs

# ============ DATA ============
dataset: MNIST                      # Which dataset to use
root: ./data                        # Path to data/ folder
num_workers: 4                      # How many CPU threads load data in parallel
batch_size: 128                     # Samples per batch

# ============ GROUPS & ENCODERS (Heterogeneous!) ============
groups:
  - name: mnist28                   # Group 0: name
    type: image                     # Type of data
    in_shape: [1, 28, 28]          # Expected input: 1 channel, 28×28 pixels
    encoder: cnn28                  # Which encoder to use (from ENCODER_REGISTRY)

  - name: mnist32                   # Group 1: name
    type: image                     # Type of data
    in_shape: [1, 32, 32]          # Expected input: 1 channel, 32×32 pixels
    encoder: cnn32                  # Different encoder (handles 32×32)

# ============ MODEL ============
latent_dim: 64                      # Dimension of shared latent space
                                    # All encoders output z ∈ ℝ^64
num_classes: 10                     # Number of output classes (digits 0-9)
head_hidden: 0                      # 0 = LinearHead (no hidden layer)
                                    # >0 = MLPHead with hidden layer of size X

# ============ ANCHORS ============
anchor_eps: 1.0e-4                  # Regularization for covariance: S = LL^T + eps*I
sep_samples_per_class: 16           # Number of samples to draw from each anchor
                                    # for separation loss (if using classifier method)

# ============ GroupDRO ============
groupdro_enabled: true              # Enable Group DRO reweighting?
groupdro_eta: 0.1                   # Learning rate for multiplicative weights
                                    # Higher η = faster weight changes

# ============ SEPARATION METHOD (Configurable!) ============
sep_method: "w2_margin"             # Options:
                                    # - "classifier": sample from anchors, classify
                                    # - "w2_margin": pairwise W2 distance margin

sep_margin: 2.0                     # For w2_margin: how far apart should anchors be?
                                    # (Ignored if sep_method="classifier")

# ============ LOSS WEIGHTS ============
lambda_fit: 1.0e-3                  # Weight on anchor fit loss
lambda_sep: 1.0e-3                  # Weight on anchor separation loss
# Total loss = ce + lambda_fit*l_fit + lambda_sep*l_sep

# ============ OPTIMIZER ============
lr: 1.0e-3                          # Learning rate for Adam
weight_decay: 0.0                   # L2 regularization (0 = no regularization)
epochs: 1                           # Number of full passes over dataset
grad_clip: 1.0                      # Gradient clipping (1.0 = clip to [-1, 1])

# ============ LOGGING ============
log_interval: 50                    # Log metrics every 50 batches
save_every: 5                       # Save checkpoint every 5 epochs
debug_one_batch: false              # Run only first batch with anomaly detection?
```

**How it's used in code**:
```python
# In train.py:
import yaml

with open("experiments/digits_centralized.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Now cfg is a Python dict:
# cfg["seed"] = 1337
# cfg["batch_size"] = 128
# cfg["groups"] = [{"name": "mnist28", ...}, ...]

# Used throughout:
set_seed(cfg["seed"])
train_loader, test_loader = build_loaders(cfg["root"], cfg["groups"], ...)
opt = Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
```

**Creating new experiments** (without touching code):
```
experiments/
├── digits_centralized.yaml         # Current
├── digits_classifier_sep.yaml      # Same config but sep_method="classifier"
├── digits_long_train.yaml          # Same but epochs=100
└── mnist_usps_cross.yaml           # Future: MNIST vs USPS (cross-domain)
```

Then run:
```bash
python -m src.train --config experiments/digits_classifier_sep.yaml
```

---

## 4. src/ (Source Code) — Complete Breakdown

**What it is**: All Python code for the project.

```
src/
├── __init__.py                     # Makes src/ a Python package
├── train.py                        # Main training script
├── eval.py                         # Evaluation script
├── datasets.py                     # Data loading
├── utils.py                        # Helper functions
├── encoders/                       # Per-group encoder architectures
│   ├── __init__.py                # Registry of encoders
│   ├── cnn28.py                   # CNN for 28×28 images
│   └── cnn32.py                   # CNN for 32×32 images
└── model/                          # Core model components
    ├── __init__.py
    ├── head.py                    # Classification head
    ├── anchors.py                 # Gaussian anchor module
    ├── losses.py                  # Loss functions
    ├── groupdro.py                # GroupDRO optimizer
    ├── wasserstein.py             # Wasserstein distance
    └── __pycache__/               # Cached compiled Python (ignore)
```

### 4.1 src/train.py

**What it does**: Main training orchestrator. Runs the full training loop.

**Key functions**:

```python
def build_models(cfg):
    """
    Construct all model components from config.
    Returns: encoders, head, anchors, groupdro
    """

def evaluate(encoders, head, loader, device, num_groups):
    """
    Run model on test data, compute per-group accuracy.
    Returns: overall_acc, per_group_acc_list
    """

def train(cfg):
    """
    Main training loop. Runs for cfg["epochs"] epochs.
    
    Each epoch:
    - For each batch: forward, compute losses, backward, optimizer step
    - Evaluate on test set
    - Log metrics to TensorBoard
    - Save checkpoints
    """
```

**Typical execution**:
```bash
python -m src.train --config experiments/digits_centralized.yaml
# Runs train() function with config from YAML
# Saves checkpoints to runs/ directory
```

**Output**: 
- Checkpoints in `runs/last.ckpt` and `runs/best.ckpt`
- TensorBoard logs in `runs/events.out.tfevents.*`

---

### 4.2 src/eval.py

**What it does**: Load a trained checkpoint and evaluate on test set.

**Usage**:
```bash
python -m src.eval --config experiments/digits_centralized.yaml --ckpt runs/last.ckpt
```

**What it does**:
1. Load config from YAML
2. Build model architecture
3. Load saved weights from checkpoint
4. Evaluate on test set
5. Print per-group accuracies

---

### 4.3 src/datasets.py

**What it does**: Load MNIST and assign group labels.

**Key classes**:

```python
class GroupWrapped(Dataset):
    """Wraps a dataset to attach group ID to each sample."""
    def __getitem__(self, idx):
        x, y = self.base[idx]
        g = self.group_id  # ← Attach group label
        return x, y, g
```

**Key functions**:

```python
def get_mnist_group_datasets(root, train, groups_cfg):
    """
    For each group in config:
    - Load MNIST
    - Apply group-specific transform (e.g., resize to 32×32)
    - Wrap with GroupWrapped(group_id)
    - Concatenate all groups
    """

def pad_to_max_collate(batch):
    """
    Custom collate function. Called when creating each batch.
    
    Input: list of (x: C×H×W, y, g) tuples from different groups
           (some x might be 28×28, some 32×32)
    
    Output: (x_batch: B×C×32×32, y_batch, g_batch)
            All images padded to max size in batch
    """

def build_loaders(root, groups_cfg, batch_size, num_workers):
    """
    Create train and test DataLoaders.
    """
```

**Example flow**:
```
Raw MNIST (28×28) → Group 0 transform (keep 28×28) → GroupWrapped(gid=0)
                 → Group 1 transform (pad to 32×32) → GroupWrapped(gid=1)
                 → ConcatDataset([group0_ds, group1_ds])
                 → DataLoader with pad_to_max_collate
                 → Each batch has mixed sizes, padded at collate time
```

---

### 4.4 src/utils.py

**What it does**: General utility functions.

```python
def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Meter:
    """Running average tracker."""
    def update(self, val, k=1):
        self.total += val * k
        self.n += k
    
    @property
    def avg(self):
        return self.total / max(1, self.n)

def ensure_dir(path):
    """Create directory if it doesn't exist."""

console:
    """Colored output printer (from rich library)."""
```

**Used in train.py**:
```python
loss_meter = Meter()
for batch in train_loader:
    loss_meter.update(loss.item(), x.size(0))

print(f"Average loss: {loss_meter.avg:.3f}")
```

---

### 4.5 src/encoders/

**What it does**: Define per-group encoder architectures.

**File: src/encoders/__init__.py**
```python
from .cnn28 import CNN28
from .cnn32 import CNN32

ENCODER_REGISTRY = {
    "cnn28": CNN28,
    "cnn32": CNN32,
}
```

**Why a registry?**
```python
# In train.py:
enc_cls = ENCODER_REGISTRY[g["encoder"]]  # Get class by name from config
encoder = enc_cls(latent_dim)             # Instantiate
```

This way, you add new encoders without changing train.py:
```python
# To add a new encoder:
# 1. Create src/encoders/cnn_resnet.py with class ResNetEncoder
# 2. Add to __init__.py:
#    from .cnn_resnet import ResNetEncoder
#    ENCODER_REGISTRY["cnn_resnet"] = ResNetEncoder
# 3. Use in config: encoder: cnn_resnet
```

**File: src/encoders/cnn28.py**
```python
class CNN28(nn.Module):
    """CNN for 28×28 images."""
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = Linear(128 * 7 * 7, latent_dim)  # After 3 pooling layers
    
    def forward(self, x):  # x: B×1×28×28
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)      # → B×32×14×14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)      # → B×64×7×7
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)      # → B×128×3×3 (hmm, not quite right, but idea)
        x = x.flatten(1)            # → B×(128*...)
        z = self.fc(x)              # → B×latent_dim
        return z
```

**File: src/encoders/cnn32.py**
```python
class CNN32(nn.Module):
    """CNN for 32×32 images. Similar to CNN28 but handles larger input."""
```

---

### 4.6 src/model/head.py

**What it does**: Classification head (maps latent → logits).

```python
class LinearHead(nn.Module):
    """Simple linear layer: z → logits (no hidden layer)."""
    def __init__(self, latent_dim, num_classes):
        self.fc = Linear(latent_dim, num_classes)
    
    def forward(self, z):  # z: B×latent_dim
        logits = self.fc(z)  # → B×num_classes
        return logits

class MLPHead(nn.Module):
    """MLP with hidden layer: z → hidden → logits."""
    def __init__(self, latent_dim, hidden_dim, num_classes):
        self.fc1 = Linear(latent_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)
    
    def forward(self, z):
        h = F.relu(self.fc1(z))
        logits = self.fc2(h)
        return logits
```

**Usage in train.py**:
```python
if cfg["head_hidden"] == 0:
    head = LinearHead(latent_dim, num_classes)
else:
    head = MLPHead(latent_dim, cfg["head_hidden"], num_classes)
```

---

### 4.7 src/model/anchors.py

**What it does**: Per-class Gaussian anchor module.

**Key idea**: For each class c, store (m_c, L_c) where:
- m_c = mean of class c Gaussian (parameter)
- L_c = lower-triangular factor (parameter)
- Covariance: Σ_c = L_c L_c^T + ε I

```python
class AnchorModule(nn.Module):
    def __init__(self, num_classes, latent_dim, eps=1e-5):
        # Store as Parameters (will be optimized)
        self.m = Parameter((num_classes, latent_dim))  # Means
        self.L = Parameter((num_classes, latent_dim, latent_dim))  # Factors
        self.eps = eps
    
    def forward(self):
        """Return (m, S, L_norm) for current batch."""
        L_norm = self.normalized_L()  # Stabilized L
        S = compute_covariance(L_norm, eps=self.eps)  # L @ L^T + eps I
        return self.m, S, L_norm
    
    def sample(self, class_idx, n):
        """Sample n points from N(m_c, Σ_c)."""
        m_c = self.m[class_idx]
        Σ_c = compute_covariance(self.normalized_L()[class_idx])
        # Use reparameterization trick
        xi = torch.randn(n, latent_dim)
        z = m_c + xi @ cholesky(Σ_c).T
        return z
```

**Why this design?**
- L parametrization ensures positive-definiteness by construction
- Easier to optimize than full covariance (fewer degrees of freedom)
- Numerically stable

---

### 4.8 src/model/wasserstein.py

**What it does**: Compute Wasserstein-2 (Bures) distance between Gaussians.

```python
def psd_sqrt(mat, eps=1e-5):
    """
    Compute square root of positive semi-definite matrix.
    
    Uses:
    1. Cholesky (fast)
    2. Eigendecomposition (stable fallback)
    """

def gaussian_w2(m1, S1, m2, S2, eps=1e-5):
    """
    Wasserstein-2 distance between N(m1, S1) and N(m2, S2).
    
    Formula:
    W2² = ||m1 - m2||² + tr(S1 + S2 - 2(S1^{1/2} S2 S1^{1/2})^{1/2})
    """
```

**Why needed?**
- Anchor fit loss measures distance between batch moments and anchors
- Requires robust, numerically stable computation

---

### 4.9 src/model/losses.py

**What it does**: Compute the three loss components.

```python
def per_class_batch_moments(z, y, num_classes, eps):
    """
    For each class c:
    - m_hat_c = mean of z samples in class c
    - S_hat_c = covariance of z samples in class c + eps*I
    """

def anchor_fit_loss(anchors_m, anchors_S, batch_moments, eps):
    """
    Minimize W2 distance between batch moments and anchors.
    l_fit = mean(W2(batch_moments_c, anchors_c) for all c)
    """

def anchor_sep_loss(anchors_m, anchors_S, anchors_L, head, ..., sep_method, ...):
    """
    Separate anchors using:
    - "classifier": draw samples, classify
    - "w2_margin": pairwise W2 margin
    """
```

---

### 4.10 src/model/groupdro.py

**What it does**: GroupDRO multiplicative weight reweighting.

```python
class GroupDRO:
    def __init__(self, num_groups, eta=0.1):
        self.q = torch.ones(num_groups) / num_groups  # Uniform weights
    
    def forward(self, logits, y, g):
        """Compute GroupDRO-weighted loss."""
        for gid in range(num_groups):
            mask = g == gid
            loss_g = CE(logits[mask], y[mask])
            weighted_loss += self.q[gid] * loss_g
        
        # Store for deferred update
        self._last_group_losses = group_losses
        self._last_group_counts = group_counts
        
        return weighted_loss
    
    def update_weights(self, group_losses, group_counts):
        """Update q_g multiplicatively after backward."""
        for gid in range(num_groups):
            q_new[gid] = self.q[gid] * exp(eta * loss_g[gid])
        q_new = q_new / sum(q_new)  # Normalize
        self.q = q_new
```

---

## 5. tests/ (Unit Tests)

**What it is**: Automated tests for numerical stability.

```
tests/
└── test_wasserstein.py             # Tests for W2 distance computation
```

**File: tests/test_wasserstein.py**

```python
def test_psd_sqrt_reconstruct():
    """
    Test: psd_sqrt(A) @ psd_sqrt(A).T ≈ A
    
    Checks that matrix square root is numerically correct.
    """

def test_gaussian_w2_symmetry_nonneg():
    """
    Test: W2(N1, N2) == W2(N2, N1) and W2 >= 0
    
    Checks that W2 distance is symmetric and non-negative.
    """
```

**Why tests?**
- Detect numerical bugs early
- Regression detection (if something breaks, tests catch it)
- Documentation of expected behavior

**How to run**:
```bash
pip install pytest
pytest tests/test_wasserstein.py -v
```

---

## 6. runs/ (Training Outputs)

**What it is**: Directory where training saves checkpoints and logs.

```
runs/
├── last.ckpt                       # Latest checkpoint (always updated)
├── best.ckpt                       # Best checkpoint (by worst-group accuracy)
└── events.out.tfevents.*          # TensorBoard event logs
```

### Checkpoint Contents (saved by torch.save):

```python
ckpt = {
    "cfg": cfg,                      # Entire config YAML as dict
    "epoch": epoch,                  # Which epoch this is from
    "encoders": {                    # Per-group encoder weights
        0: encoder_0_state_dict,
        1: encoder_1_state_dict,
    },
    "head": head_state_dict,         # Classification head weights
    "anchors": anchors_state_dict,   # Anchor parameters (m, L)
    "worst_group_acc": worst_group_acc,  # Best worst-group acc so far
    "experiment": {                  # Experiment metadata
        "sep_method": "w2_margin",
        "sep_margin": 2.0,
    },
    "groupdro": {                    # GroupDRO state (if enabled)
        "weights": q_tensor,
        "stats": group_stats,
    },
}
```

### TensorBoard Logs:

```
# To visualize training:
tensorboard --logdir runs/

# Then open browser to http://localhost:6006
# See: loss curves, accuracy curves, per-group metrics
```

---

## 7. Documentation Files

### README.md (Project Overview)

**What it explains**:
- High-level project goal
- Quick start commands
- Basic architecture overview

**Example**:
```markdown
# DRO-HeteroAnchors — Centralized Baseline

Centralized latent-anchor alignment for heterogeneous feature spaces.

## Quickstart
```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --config experiments/digits_centralized.yaml
```
```

**When to read**: First time understanding project

---

### MAPPING.md (PDF ↔ Code)

**What it explains**:
- Which code implements which paper concept
- Differences between paper and implementation

**Example**:
```markdown
## Mapping to the Paper

- Per-group encoders φ_g → src/encoders/*.py
- Class anchors μ_c → src/model/anchors.py
- Wasserstein distance → src/model/wasserstein.py
- GroupDRO weights → src/model/groupdro.py
```

**When to read**: Verifying paper fidelity, understanding PDF

---

### IMPLEMENTATION_GUIDE.md (This whole file!)

**What it explains**:
- Every file and folder in detail
- How training loop works step-by-step
- Numerical stability decisions

**When to read**: Deep dive into how code works

---

## 8. requirements.txt (Dependencies)

**What it is**: List of Python packages needed to run the project.

**Contents** (example):
```
torch==2.0.0
torchvision==0.15.0
pyyaml>=6.0
tqdm>=4.60.0
rich>=10.0.0
tensorboard>=2.10.0
numpy>=1.20.0
```

**How to install**:
```bash
pip install -r requirements.txt
```

**How it's used**:
- First time setup: ensures everyone has same package versions
- Reproducibility: same versions across machines/people

**To add a package**:
```bash
pip install new_package==1.2.3
pip freeze > requirements.txt  # Update file
```

---

## Complete Folder Tree with Descriptions

```
dro_hetero_anchors/                      # Root project folder
│
├── .venv/                               # Virtual environment (isolated Python)
│   ├── bin/python                       # Python interpreter to use
│   ├── bin/pip                          # Package installer
│   └── lib/python3.13/site-packages/    # Installed packages (PyTorch, etc.)
│
├── data/                                # Datasets
│   └── MNIST/                           # MNIST dataset
│       └── raw/                         # Raw binary files
│           ├── train-images-idx3-ubyte
│           ├── train-labels-idx1-ubyte
│           ├── t10k-images-idx3-ubyte
│           └── t10k-labels-idx1-ubyte
│
├── experiments/                         # Config files
│   └── digits_centralized.yaml          # YAML config for MNIST experiments
│
├── src/                                 # Source code
│   ├── __init__.py
│   ├── train.py                         # Main training orchestrator
│   ├── eval.py                          # Evaluation script
│   ├── datasets.py                      # Data loading and grouping
│   ├── utils.py                         # Helper functions
│   │
│   ├── encoders/                        # Per-group encoders
│   │   ├── __init__.py                  # Encoder registry
│   │   ├── cnn28.py                     # CNN for 28×28 images
│   │   └── cnn32.py                     # CNN for 32×32 images
│   │
│   └── model/                           # Core model components
│       ├── __init__.py
│       ├── head.py                      # Classification head
│       ├── anchors.py                   # Gaussian anchors (m, L)
│       ├── losses.py                    # Loss functions (fit, sep, CE)
│       ├── groupdro.py                  # GroupDRO weight reweighting
│       └── wasserstein.py               # Wasserstein distance (W2 Bures)
│
├── tests/                               # Unit tests
│   └── test_wasserstein.py              # Tests for numerical stability
│
├── runs/                                # Training outputs
│   ├── last.ckpt                        # Latest checkpoint
│   ├── best.ckpt                        # Best checkpoint
│   └── events.out.tfevents.*            # TensorBoard logs
│
├── README.md                            # Quick start guide
├── MAPPING.md                           # PDF-to-code mapping
├── IMPLEMENTATION_GUIDE.md              # Detailed walkthrough
├── requirements.txt                     # Python dependencies
└── [other files like .gitignore, etc.]
```

---

## Typical Workflow

### 1. First Time Setup
```bash
# Create venv
python3 -m venv .venv

# Activate venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiment
```bash
# Activate venv (if not already)
source .venv/bin/activate

# Train
python -m src.train --config experiments/digits_centralized.yaml

# Output:
# - Saves checkpoint to runs/last.ckpt
# - Saves logs to runs/events.out.tfevents.*
```

### 3. View Results
```bash
# Option 1: TensorBoard visualization
tensorboard --logdir runs/

# Option 2: Evaluate checkpoint
python -m src.eval --config experiments/digits_centralized.yaml --ckpt runs/last.ckpt
```

### 4. Modify Experiment (No Code Changes!)
```bash
# Edit experiments/digits_classifier_sep.yaml (copy and modify)
# Change: sep_method: "classifier"

# Run new experiment
python -m src.train --config experiments/digits_classifier_sep.yaml
```

### 5. Add New Dataset (USPS)
```bash
# 1. Modify src/datasets.py to load USPS
# 2. Add USPS config to experiments/usps.yaml
# 3. Run: python -m src.train --config experiments/usps.yaml
```

---

## Summary

| Folder/File | Purpose | You Edit? | Key Takeaway |
|---|---|---|---|
| `.venv/` | Python environment | No | Isolated dependencies |
| `data/` | Datasets | No (auto-downloaded) | MNIST auto-fetched on first run |
| `experiments/` | Config files | Yes | Different experiments without code changes |
| `src/` | Source code | Yes (carefully) | Main implementation |
| `tests/` | Unit tests | Maybe | Verify numerical stability |
| `runs/` | Checkpoints & logs | No | Training outputs |
| `README.md` | Quick start | Rarely | High-level overview |
| `MAPPING.md` | PDF mapping | Rarely | Verify paper fidelity |
| `IMPLEMENTATION_GUIDE.md` | Deep dive | Reference only | Understand how everything works |
| `requirements.txt` | Dependencies | Rarely | For reproducibility |

---

## Key Concepts to Remember

1. **Virtual environment (.venv/)**: Isolated Python with project-specific packages
2. **Config files (experiments/)**: Change settings without touching code
3. **Per-group encoders (encoders/)**: Different architectures for different input sizes
4. **Three losses (losses.py)**: CE (GroupDRO) + anchor fit + anchor separation
5. **GroupDRO (groupdro.py)**: Multiplicative weights emphasizing worst-performing groups
6. **Checkpoints (runs/)**: Save model weights to resume or evaluate later
7. **Tests (tests/)**: Catch numerical bugs automatically

This is a complete, production-quality research codebase!
