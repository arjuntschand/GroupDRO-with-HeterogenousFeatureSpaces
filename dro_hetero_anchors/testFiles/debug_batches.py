#!/usr/bin/env python3
"""(Moved to testFiles/) Debug script to check batch class distribution in USPS diagnostic.

Purpose: Inspect early training batches for class presence to detect skew-induced omissions.
"""

import sys
from pathlib import Path
# Ensure package import works when run from this folder
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from torch.utils.data import DataLoader
import numpy as np
from src.datasets import build_skewed_mnist_usps_loaders

def debug_batch_distribution():
    """Check if classes 1-4 are systematically missing from batches"""
    
    # Replicate the exact settings from usps_only_diagnostic.yaml
    np.random.seed(1337)
    torch.manual_seed(1337)
    
    data_root = str(Path(__file__).resolve().parents[2] / 'datasets')
    train_loader, test_loader = build_skewed_mnist_usps_loaders(
        root=data_root, 
        batch_size=128, 
        num_workers=4,
        mnist_size=1,
        usps_size=7291,
        mnist_majority=[0, 1, 2, 3, 4],
        usps_majority=[0, 1, 2, 3, 4],
        seed=1337
    )
    
    print("Analyzing batch class distributions...")
    
    class_batch_counts = {c: 0 for c in range(10)}
    batches_with_class = {c: 0 for c in range(10)}
    total_batches = 0
    
    for batch_idx, (x_batch, y_batch, g_batch) in enumerate(train_loader):
        total_batches += 1
        
        # Count classes in this batch
        batch_class_counts = {c: 0 for c in range(10)}
        for c in range(10):
            count = (y_batch == c).sum().item()
            batch_class_counts[c] = count
            if count > 0:
                batches_with_class[c] += 1
            class_batch_counts[c] += count
            
        # Print first few batches to see the pattern
        if batch_idx < 5:
            print(f"Batch {batch_idx}: {batch_class_counts}")
            
        # Check for problem pattern
        if batch_class_counts[1] == 0 and batch_class_counts[2] == 0 and batch_class_counts[3] == 0 and batch_class_counts[4] == 0:
            if batch_idx < 10:  # Only print first 10 to avoid spam
                print(f"  ⚠️  Batch {batch_idx}: Classes 1-4 all missing!")
    
    print(f"\nAnalyzed {total_batches} total batches")
    print(f"Total samples per class across all batches:")
    for c in range(10):
        print(f"  Class {c}: {class_batch_counts[c]} samples in {batches_with_class[c]}/{total_batches} batches")
    
    print(f"\nBatches where each class appears:")
    for c in range(10):
        pct = 100.0 * batches_with_class[c] / total_batches
        print(f"  Class {c}: {batches_with_class[c]}/{total_batches} batches ({pct:.1f}%)")

if __name__ == "__main__":
    debug_batch_distribution()