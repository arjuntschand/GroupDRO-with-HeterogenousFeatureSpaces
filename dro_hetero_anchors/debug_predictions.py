import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import yaml
from collections import Counter

# Load checkpoint
ckpt_path = Path('./runs/usps_only_fixed/best.ckpt')
ckpt = torch.load(ckpt_path, map_location='cpu')
cfg = ckpt['cfg']

# Rebuild models
from src.encoders.cnn28 import CNN28
from src.model.head import LinearHead

device = torch.device('cpu')
enc = CNN28(latent_dim=64).to(device)
enc.load_state_dict(ckpt['encoders'][1])  # Group 1 is USPS
enc.eval()

head = LinearHead(64, 10).to(device)
head.load_state_dict(ckpt['head'])
head.eval()

# Load USPS test data (complement of training)
usps_full = datasets.USPS(root='./data', train=True, transform=transforms.ToTensor())

# Get training indices from config sampling
import numpy as np
np.random.seed(1337)
usps_targets = np.array(usps_full.targets)

# Replicate sampling logic to get same train/test split
usps_majority = [0, 1, 2, 3, 4]
usps_size = 7291

class_indices = {c: np.where(usps_targets == c)[0] for c in range(10)}
minority_classes = [5, 6, 7, 8, 9]

num_majority = int(0.7 * usps_size)
num_minority = usps_size - num_majority

sampled = []
for c in usps_majority:
    available = class_indices[c]
    n_per_class = num_majority // len(usps_majority)
    sampled.extend(np.random.choice(available, size=min(n_per_class, len(available)), replace=False))

remaining = num_majority - len(sampled)
if remaining > 0:
    all_majority_idxs = np.concatenate([class_indices[c] for c in usps_majority])
    sampled.extend(np.random.choice(all_majority_idxs, size=remaining, replace=True))

for c in minority_classes:
    available = class_indices[c]
    n_per_class = num_minority // len(minority_classes)
    sampled.extend(np.random.choice(available, size=min(n_per_class, len(available)), replace=False))

remaining = num_minority - (len(sampled) - num_majority)
if remaining > 0:
    all_minority_idxs = np.concatenate([class_indices[c] for c in minority_classes])
    sampled.extend(np.random.choice(all_minority_idxs, size=remaining, replace=True))

train_indices = list(set(sampled))[:usps_size]

# Test indices are complement
all_indices = set(range(len(usps_full)))
test_indices = list(all_indices - set(train_indices))

test_ds = Subset(usps_full, test_indices)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

# Run inference and collect predictions per class
predictions_by_class = {c: [] for c in range(10)}
true_labels_by_class = {c: [] for c in range(10)}

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        z = enc(x)
        logits = head(z)
        preds = logits.argmax(dim=1)
        
        for i in range(len(y)):
            true_class = int(y[i].item())
            pred_class = int(preds[i].item())
            predictions_by_class[true_class].append(pred_class)
            true_labels_by_class[true_class].append(true_class)

# Print confusion for classes 1-4
print("Predictions for USPS test samples (classes 1-4):\n")
for c in [1, 2, 3, 4]:
    preds = predictions_by_class[c]
    if len(preds) == 0:
        print(f"Class {c}: NO TEST SAMPLES")
        continue
    counter = Counter(preds)
    total = len(preds)
    print(f"Class {c} ({total} test samples):")
    for pred_c in sorted(counter.keys()):
        count = counter[pred_c]
        pct = 100.0 * count / total
        print(f"  Predicted as {pred_c}: {count}/{total} ({pct:.1f}%)")
    correct = counter[c]
    acc = 100.0 * correct / total
    print(f"  Accuracy: {acc:.1f}%\n")

# Also print for working classes
print("\nPredictions for USPS test samples (classes 0, 5-9):\n")
for c in [0, 5, 6, 7, 8, 9]:
    preds = predictions_by_class[c]
    if len(preds) == 0:
        print(f"Class {c}: NO TEST SAMPLES")
        continue
    counter = Counter(preds)
    total = len(preds)
    correct = counter.get(c, 0)
    acc = 100.0 * correct / total
    print(f"Class {c} ({total} samples): {acc:.1f}% accuracy")
