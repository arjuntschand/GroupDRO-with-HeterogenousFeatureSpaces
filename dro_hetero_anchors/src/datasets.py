from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
from torchvision import datasets, transforms
import ssl
import urllib.request

# Disable SSL verification for downloading datasets (Python 3.13 macOS certificate issue)
ssl._create_default_https_context = ssl._create_unverified_context

class GroupWrapped(Dataset):
    """Wraps a base dataset to attach a fixed group index to each sample."""
    def __init__(self, base: Dataset, group_id: int):
        self.base = base
        self.group_id = group_id
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        g = self.group_id
        return x, y, g

def get_mnist_group_datasets(root: str, train: bool, groups_cfg: List[dict]) -> Dataset:
    """
    Build a ConcatDataset from multiple MNIST-based groups with different transforms.
    Each group's transform can yield a different H×W; we'll pad at collate time.
    """
    ds_list: List[Dataset] = []
    for gid, g in enumerate(groups_cfg):
        _, H, W = g["in_shape"]  # declared target shape for this group's encoder
        if H == 28 and W == 28:
            tfm = transforms.ToTensor()
        else:
            if H >= 28 and W >= 28:
                pad_h = (H - 28) // 2
                pad_w = (W - 28) // 2
                tfm = transforms.Compose([transforms.Pad((pad_w, pad_h)), transforms.ToTensor()])
            else:
                tfm = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor()])
        base = datasets.MNIST(root=root, train=train, transform=tfm, download=True)
        ds_list.append(GroupWrapped(base, gid))
    return ConcatDataset(ds_list)

def pad_to_max_collate(batch):
    """
    Pads each image in the batch to the maximum H and W in this batch, then stacks.
    Batch is a list of tuples: (x: C×H×W, y, g)
    """
    xs, ys, gs = zip(*batch)
    H = max(x.shape[1] for x in xs)
    W = max(x.shape[2] for x in xs)

    x_padded = []
    for x in xs:
        h, w = x.shape[1], x.shape[2]
        pad_h = H - h
        pad_w = W - w
        # pad = (left, right, top, bottom)
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        x_padded.append(F.pad(x, pad))
    x_batch = torch.stack(x_padded, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    g = torch.tensor(gs, dtype=torch.long)
    return x_batch, y, g

def build_loaders(root: str, groups_cfg: List[dict], batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    train_ds = get_mnist_group_datasets(root, True, groups_cfg)
    test_ds = get_mnist_group_datasets(root, False, groups_cfg)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)
    return train_loader, test_loader


def build_skewed_mnist_usps_loaders(root: str, batch_size: int, num_workers: int, 
                                     mnist_size: int = 30000, usps_size: int = 1000,
                                     mnist_majority: List[int] = None, usps_majority: List[int] = None,
                                     seed: int = 1337) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test loaders with skewed MNIST and USPS datasets.
    
    Args:
        root: data root path
        batch_size: batch size
        num_workers: number of workers
        mnist_size: number of MNIST samples to use (30000)
        usps_size: number of USPS samples to use (1000)
        mnist_majority: majority class indices for MNIST (default [0,1,2,3,4])
        usps_majority: majority class indices for USPS (default [5,6,7,8,9])
        seed: random seed
    
    Returns:
        train_loader, test_loader (both use skewed datasets)
    """
    if mnist_majority is None:
        mnist_majority = [0, 1, 2, 3, 4]
    if usps_majority is None:
        usps_majority = [5, 6, 7, 8, 9]
    
    mnist_minority = [c for c in range(10) if c not in mnist_majority]
    usps_minority = [c for c in range(10) if c not in usps_majority]
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load MNIST
    mnist_full = datasets.MNIST(root=root, train=True, transform=transforms.ToTensor(), download=True)
    mnist_targets = np.array(mnist_full.targets)
    
    # Load USPS (only train split available)
    usps_full = datasets.USPS(root=root, train=True, transform=transforms.ToTensor(), download=True)
    usps_targets = np.array(usps_full.targets)
    
    # NEW: First split into train/test with balanced classes, THEN apply skew to train only
    def split_balanced_then_skew(targets, train_size, majority_classes, train_ratio=0.8):
        """
        1. Split data 80/20 train/test with EQUAL class proportions
        2. Within train set, apply skewed sampling: 80% to majority, 20% to minority
        
        Args:
            targets: array of labels
            train_size: target number of training samples
            majority_classes: which classes get 80% of train samples
            train_ratio: what fraction of TOTAL data to use for training (default 0.8)
        
        Returns:
            train_indices, test_indices
        """
        class_indices = {c: np.where(targets == c)[0] for c in range(10)}
        minority_classes = [c for c in range(10) if c not in majority_classes]
        
        # Step 1: For each class, split into train_ratio and (1-train_ratio) for test
        train_pool = {c: [] for c in range(10)}
        test_indices = []
        
        for c in range(10):
            available = class_indices[c]
            n_train = int(len(available) * train_ratio)
            np.random.shuffle(available)
            train_pool[c] = available[:n_train].tolist()
            test_indices.extend(available[n_train:].tolist())
        
        # Step 2: From train_pool, sample train_size with 80% majority, 20% minority
        num_majority = int(0.8 * train_size)
        num_minority = train_size - num_majority
        
        train_indices = []
        
        # Sample from majority classes
        for c in majority_classes:
            n_per_class = num_majority // len(majority_classes)
            available = train_pool[c]
            sampled = min(n_per_class, len(available))
            train_indices.extend(np.random.choice(available, size=sampled, replace=False).tolist())
        
        # Fill remaining majority slots
        remaining = num_majority - len(train_indices)
        if remaining > 0:
            all_majority = [idx for c in majority_classes for idx in train_pool[c]]
            if len(all_majority) > 0:
                train_indices.extend(np.random.choice(all_majority, size=remaining, replace=True).tolist())
        
        # Sample from minority classes
        for c in minority_classes:
            n_per_class = num_minority // len(minority_classes)
            available = train_pool[c]
            sampled = min(n_per_class, len(available))
            train_indices.extend(np.random.choice(available, size=sampled, replace=False).tolist())
        
        # Fill remaining minority slots
        current_minority = len(train_indices) - num_majority
        remaining = num_minority - current_minority
        if remaining > 0:
            all_minority = [idx for c in minority_classes for idx in train_pool[c]]
            if len(all_minority) > 0:
                train_indices.extend(np.random.choice(all_minority, size=remaining, replace=True).tolist())
        
        return train_indices[:train_size], test_indices
    
    # Sample with balanced test split, then skewed train
    mnist_train_indices, mnist_test_indices = split_balanced_then_skew(
        mnist_targets, mnist_size, mnist_majority, train_ratio=0.8)
    usps_train_indices, usps_test_indices = split_balanced_then_skew(
        usps_targets, usps_size, usps_majority, train_ratio=0.8)
    
    # Create train datasets with transforms (MNIST 28x28, USPS 16x16)
    mnist_train_tfm = transforms.ToTensor()
    usps_train_tfm = transforms.ToTensor()  # USPS is already 16x16
    
    mnist_train = datasets.MNIST(root=root, train=True, transform=mnist_train_tfm, download=True)
    usps_train = datasets.USPS(root=root, train=True, transform=usps_train_tfm, download=True)
    
    mnist_train_subset = Subset(mnist_train, mnist_train_indices)
    usps_train_subset = Subset(usps_train, usps_train_indices)
    
    # Wrap with group IDs: group 0 = MNIST 28x28, group 1 = USPS 16x16
    mnist_train_wrapped = GroupWrapped(mnist_train_subset, 0)
    usps_train_wrapped = GroupWrapped(usps_train_subset, 1)
    
    train_ds = ConcatDataset([mnist_train_wrapped, usps_train_wrapped])
    
    # Test sets already computed by split_balanced_then_skew
    mnist_test_subset = Subset(mnist_train, mnist_test_indices)
    usps_test_subset = Subset(usps_train, usps_test_indices)
    
    mnist_test_wrapped = GroupWrapped(mnist_test_subset, 0)
    usps_test_wrapped = GroupWrapped(usps_test_subset, 1)
    
    test_ds = ConcatDataset([mnist_test_wrapped, usps_test_wrapped])
    
    # Build loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)
    
    return train_loader, test_loader


def build_skewed_mnist_usps_mnist32_loaders(
    root: str,
    batch_size: int,
    num_workers: int,
    mnist28_size: int = 15000,
    usps_size: int = 2000,
    mnist32_size: int = 15000,
    mnist28_majority: Optional[List[int]] = None,
    usps_majority: Optional[List[int]] = None,
    mnist32_majority: Optional[List[int]] = None,
    seed: int = 1337,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/test loaders for 3 groups: MNIST 28x28, USPS 16x16, MNIST 32x32.
    Uses the FIXED split_balanced_then_skew approach for proper train/test splitting.
    """
    if mnist28_majority is None:
        mnist28_majority = [0, 1, 2, 3, 4]
    if usps_majority is None:
        usps_majority = [5, 6, 7, 8, 9]
    if mnist32_majority is None:
        mnist32_majority = [0, 1, 2, 3, 4]

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load base datasets
    mnist_full = datasets.MNIST(root=root, train=True, transform=transforms.ToTensor(), download=True)
    usps_full = datasets.USPS(root=root, train=True, transform=transforms.ToTensor(), download=True)

    # Reuse the split_balanced_then_skew function from the 2-group loader
    # (Copy it here as a nested function)
    def split_balanced_then_skew(targets, train_size, majority_classes, train_ratio=0.8):
        """Split data 80/20 train/test with balanced classes, then apply skew to train only"""
        class_indices = {c: np.where(targets == c)[0] for c in range(10)}
        minority_classes = [c for c in range(10) if c not in majority_classes]
        
        # Step 1: For each class, split into train_ratio and (1-train_ratio) for test
        train_pool = {c: [] for c in range(10)}
        test_indices = []
        
        for c in range(10):
            available = class_indices[c]
            n_train = int(len(available) * train_ratio)
            np.random.shuffle(available)
            train_pool[c] = available[:n_train].tolist()
            test_indices.extend(available[n_train:].tolist())
        
        # Step 2: From train_pool, sample train_size with 80% majority, 20% minority
        num_majority = int(0.8 * train_size)
        num_minority = train_size - num_majority
        
        train_indices = []
        
        # Sample from majority classes
        for c in majority_classes:
            n_per_class = num_majority // len(majority_classes)
            available = train_pool[c]
            sampled = min(n_per_class, len(available))
            train_indices.extend(np.random.choice(available, size=sampled, replace=False).tolist())
        
        # Fill remaining majority slots
        remaining = num_majority - len(train_indices)
        if remaining > 0:
            all_majority = [idx for c in majority_classes for idx in train_pool[c]]
            if len(all_majority) > 0:
                train_indices.extend(np.random.choice(all_majority, size=remaining, replace=True).tolist())
        
        # Sample from minority classes
        for c in minority_classes:
            n_per_class = num_minority // len(minority_classes)
            available = train_pool[c]
            sampled = min(n_per_class, len(available))
            train_indices.extend(np.random.choice(available, size=sampled, replace=False).tolist())
        
        # Fill remaining minority slots
        current_minority = len(train_indices) - num_majority
        remaining = num_minority - current_minority
        if remaining > 0:
            all_minority = [idx for c in minority_classes for idx in train_pool[c]]
            if len(all_minority) > 0:
                train_indices.extend(np.random.choice(all_minority, size=remaining, replace=True).tolist())
        
        return train_indices, test_indices

    # Use the fixed split_balanced_then_skew for each group
    # For MNIST, we need to split the total (mnist28_size + mnist32_size) and then divide
    total_mnist_size = mnist28_size + mnist32_size
    
    # Get MNIST train/test splits using balanced approach
    mnist_train_indices, mnist_test_indices = split_balanced_then_skew(
        targets=np.array(mnist_full.targets),
        train_size=total_mnist_size,
        majority_classes=mnist28_majority,  # Use mnist28_majority as default for splitting
    )
    
    # Split MNIST train/test between 28x28 and 32x32 proportionally
    prop_28 = mnist28_size / total_mnist_size
    n_train_28 = int(round(prop_28 * len(mnist_train_indices)))
    n_test_28 = int(round(prop_28 * len(mnist_test_indices)))
    
    np.random.shuffle(mnist_train_indices)
    np.random.shuffle(mnist_test_indices)
    
    mnist28_train_idx = mnist_train_indices[:n_train_28]
    mnist32_train_idx = mnist_train_indices[n_train_28:]
    mnist28_test_idx = mnist_test_indices[:n_test_28]
    mnist32_test_idx = mnist_test_indices[n_test_28:]
    
    # Get USPS train/test splits
    usps_train_indices, usps_test_indices = split_balanced_then_skew(
        targets=np.array(usps_full.targets),
        train_size=usps_size,
        majority_classes=usps_majority,
    )

    # Create train datasets with transforms
    mnist28_tfm = transforms.ToTensor()
    mnist32_tfm = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])  # 28 -> 32 by padding
    usps_tfm = transforms.ToTensor()

    mnist_train_28 = datasets.MNIST(root=root, train=True, transform=mnist28_tfm, download=True)
    mnist_train_32 = datasets.MNIST(root=root, train=True, transform=mnist32_tfm, download=True)
    usps_train = datasets.USPS(root=root, train=True, transform=usps_tfm, download=True)

    # Create subsets
    mnist28_train_subset = Subset(mnist_train_28, mnist28_train_idx)
    mnist32_train_subset = Subset(mnist_train_32, mnist32_train_idx)
    usps_train_subset = Subset(usps_train, usps_train_indices)

    mnist28_test_subset = Subset(mnist_train_28, mnist28_test_idx)
    mnist32_test_subset = Subset(mnist_train_32, mnist32_test_idx)
    usps_test_subset = Subset(usps_train, usps_test_indices)

    # Wrap groups: 0=mnist28, 1=usps16, 2=mnist32
    train_ds = ConcatDataset([
        GroupWrapped(mnist28_train_subset, 0),
        GroupWrapped(usps_train_subset, 1),
        GroupWrapped(mnist32_train_subset, 2),
    ])

    test_ds = ConcatDataset([
        GroupWrapped(mnist28_test_subset, 0),
        GroupWrapped(usps_test_subset, 1),
        GroupWrapped(mnist32_test_subset, 2),
    ])

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)

    return train_loader, test_loader

