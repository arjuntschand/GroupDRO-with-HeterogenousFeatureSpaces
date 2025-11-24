from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
from torchvision import datasets, transforms
import ssl

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
                                     seed: int = 1337,
                                     majority_frac: float = 0.8) -> Tuple[DataLoader, DataLoader]:
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
    
    # Shared helper split function (defined once below)
    
    # Sample with balanced test split, then skewed train
    mnist_train_indices, mnist_test_indices = split_balanced_then_skew(
        mnist_targets, mnist_size, mnist_majority, train_ratio=0.8, majority_frac=majority_frac)
    usps_train_indices, usps_test_indices = split_balanced_then_skew(
        usps_targets, usps_size, usps_majority, train_ratio=0.8, majority_frac=majority_frac)
    
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
    majority_frac: float = 0.8,
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

    # Use shared helper defined below

    # Use the fixed split_balanced_then_skew for each group
    # For MNIST, we need to split the total (mnist28_size + mnist32_size) and then divide
    total_mnist_size = mnist28_size + mnist32_size
    
    # Get MNIST train/test splits using balanced approach
    mnist_train_indices, mnist_test_indices = split_balanced_then_skew(
        targets=np.array(mnist_full.targets),
        train_size=total_mnist_size,
        majority_classes=mnist28_majority,  # Use mnist28_majority as default for splitting
        majority_frac=majority_frac,
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
        majority_frac=majority_frac,
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


def build_mnist_usps_balanced_subset_loaders(
    root: str,
    batch_size: int,
    num_workers: int,
    mnist_total_size: int = 200,
    usps_total_size: int = 6000,
    train_frac: float = 0.8,
    seed: int = 1337,
    mnist_use_full_pool: bool = False,
    usps_use_full_pool: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build MNIST + USPS loaders from *balanced per-class subsets* then 80/20 split.

    Enhancements vs earlier version:
    - Optional concatenation of official train+test splits as the sampling pool ("full pool")
      to reduce per-class scarcity for large uniform requests (e.g. USPS 600/class).
    - Maintains guarantee of at least one test example per class when per-class subset > 1.

    Args:
        root: dataset directory
        batch_size: batch size
        num_workers: dataloader workers
        mnist_total_size: total MNIST subset size (train+test combined)
        usps_total_size: total USPS subset size (train+test combined)
        train_frac: fraction of each class subset allocated to train (default 0.8)
        seed: RNG seed
        mnist_use_full_pool: if True, sample MNIST from train+test concatenated pool
        usps_use_full_pool: if True, sample USPS from train+test concatenated pool
    Returns:
        train_loader, test_loader
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    def _load_pool(ds_class, use_full: bool):
        if not use_full:
            d_train = ds_class(root=root, train=True, transform=transforms.ToTensor(), download=True)
            targets = np.array(d_train.targets)
            return d_train, targets
        d_train = ds_class(root=root, train=True, transform=transforms.ToTensor(), download=True)
        d_test = ds_class(root=root, train=False, transform=transforms.ToTensor(), download=True)
        targets = np.concatenate([np.array(d_train.targets), np.array(d_test.targets)])
        concat = ConcatDataset([d_train, d_test])
        return concat, targets

    mnist_pool, mnist_targets = _load_pool(datasets.MNIST, mnist_use_full_pool)
    usps_pool, usps_targets = _load_pool(datasets.USPS, usps_use_full_pool)

    mnist_class_indices = {c: np.where(mnist_targets == c)[0] for c in range(10)}
    usps_class_indices = {c: np.where(usps_targets == c)[0] for c in range(10)}

    def sample_subset(class_indices: dict, total_size: int) -> dict:
        base = total_size // 10
        remainder = total_size - base * 10
        per_class = {c: base for c in range(10)}
        for c in range(remainder):
            per_class[c] += 1
        sampled = {}
        for c in range(10):
            idxs = class_indices[c].copy()
            np.random.shuffle(idxs)
            take = min(per_class[c], len(idxs))
            sampled[c] = idxs[:take]
        return sampled

    mnist_subset = sample_subset(mnist_class_indices, mnist_total_size)
    usps_subset = sample_subset(usps_class_indices, usps_total_size)

    def split_train_test(subset: dict, frac: float) -> Tuple[List[int], List[int]]:
        train_idx = []
        test_idx = []
        for c in range(10):
            idxs = subset[c]
            n = len(idxs)
            if n == 0:
                continue
            train_count = int(np.floor(frac * n))
            if n - train_count == 0 and n > 1:  # ensure at least one test example
                train_count -= 1
            train_idx.extend(idxs[:train_count].tolist())
            test_idx.extend(idxs[train_count:].tolist())
        return train_idx, test_idx

    mnist_train_idx, mnist_test_idx = split_train_test(mnist_subset, train_frac)
    usps_train_idx, usps_test_idx = split_train_test(usps_subset, train_frac)

    mnist_train_ds = GroupWrapped(Subset(mnist_pool, mnist_train_idx), 0)
    usps_train_ds = GroupWrapped(Subset(usps_pool, usps_train_idx), 1)
    mnist_test_ds = GroupWrapped(Subset(mnist_pool, mnist_test_idx), 0)
    usps_test_ds = GroupWrapped(Subset(usps_pool, usps_test_idx), 1)

    train_concat = ConcatDataset([mnist_train_ds, usps_train_ds])
    test_concat = ConcatDataset([mnist_test_ds, usps_test_ds])

    train_loader = DataLoader(train_concat, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)
    test_loader = DataLoader(test_concat, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)
    return train_loader, test_loader


def split_balanced_then_skew(
    targets: np.ndarray,
    train_size: int,
    majority_classes: List[int],
    train_ratio: float = 0.8,
    majority_frac: float = 0.8,
) -> Tuple[list, list]:
    """
    Shared helper: First split into train/test with balanced per-class proportions,
    then skew the TRAIN selection so that 80% of train_size comes from majority classes
    and 20% from minority classes.

    Returns train_indices, test_indices
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

    # Step 2: From train_pool, sample train_size with majority_frac majority, remainder minority
    majority_frac = float(majority_frac)
    majority_frac = max(0.0, min(1.0, majority_frac))  # clamp
    num_majority = int(majority_frac * train_size)
    num_minority = train_size - num_majority

    train_indices = []

    # Sample from majority classes
    for c in majority_classes:
        n_per_class = max(1, num_majority // max(1, len(majority_classes)))
        available = train_pool[c]
        sampled = min(n_per_class, len(available))
        if sampled > 0:
            train_indices.extend(np.random.choice(available, size=sampled, replace=False).tolist())

    # Fill remaining majority slots
    remaining = num_majority - len(train_indices)
    if remaining > 0:
        all_majority = [idx for c in majority_classes for idx in train_pool[c]]
        if len(all_majority) > 0:
            train_indices.extend(np.random.choice(all_majority, size=remaining, replace=True).tolist())

    # Sample from minority classes
    for c in minority_classes:
        n_per_class = max(1, num_minority // max(1, len(minority_classes)))
        available = train_pool[c]
        sampled = min(n_per_class, len(available))
        if sampled > 0:
            train_indices.extend(np.random.choice(available, size=sampled, replace=False).tolist())

    # Fill remaining minority slots
    current_minority = len(train_indices) - num_majority
    remaining = num_minority - current_minority
    if remaining > 0:
        all_minority = [idx for c in minority_classes for idx in train_pool[c]]
        if len(all_minority) > 0:
            train_indices.extend(np.random.choice(all_minority, size=remaining, replace=True).tolist())

    return train_indices[:train_size], test_indices


def build_usps_only_balanced_loaders(
    root: str,
    batch_size: int,
    num_workers: int,
    usps_size: int = 5000,
    seed: int = 1337,
    train_ratio: float = 0.8,  # UNUSED legacy param
    max_train_frac: float = 0.9,  # per-class cap (e.g. keep at most 90% of each class for train)
) -> Tuple[DataLoader, DataLoader]:
    """Build USPS-only loaders with a balanced train subset.

    We select exactly ``usps_size`` training samples with (near) equal per-class counts.
    Remaining samples serve as the test set. If ``usps_size`` is not divisible by 10,
    leftover samples are distributed one per class until exhausted.

    Args:
        root: dataset root directory
        batch_size: batch size
        num_workers: dataloader workers
        usps_size: number of USPS training samples desired (default 5000)
        seed: RNG seed
        train_ratio: UNUSED for now (kept for potential future extension)

    Returns:
        train_loader, test_loader
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    usps_full = datasets.USPS(root=root, train=True, transform=transforms.ToTensor(), download=True)
    targets = np.array(usps_full.targets)
    class_indices = {c: np.where(targets == c)[0] for c in range(10)}

    # Per-class maximum allowed for training (cap) and global allocation respecting usps_size
    per_class_cap = {c: int(min(len(class_indices[c]), np.floor(max_train_frac * len(class_indices[c]))))
                     for c in range(10)}
    # Ensure at least 1 test sample remains if class has >1 sample
    for c in range(10):
        if len(class_indices[c]) > 1 and per_class_cap[c] == len(class_indices[c]):
            per_class_cap[c] -= 1

    total_cap = sum(per_class_cap.values())
    effective_usps_size = min(usps_size, total_cap)

    # If requesting at least the capped total, use exact per-class caps (preserves original distribution 80/20)
    if effective_usps_size == total_cap:
        alloc = per_class_cap.copy()
    else:
        # Otherwise balanced allocation under caps to reach effective_usps_size
        base = effective_usps_size // 10
        remainder = effective_usps_size - base * 10
        alloc = {c: min(base, per_class_cap[c]) for c in range(10)}
        # Distribute remainder round-robin among classes with remaining capacity
        c_iter = 0
        while remainder > 0:
            if alloc[c_iter] < per_class_cap[c_iter]:
                alloc[c_iter] += 1
                remainder -= 1
            c_iter = (c_iter + 1) % 10
            if c_iter == 0 and all(alloc[c] == per_class_cap[c] for c in range(10)):
                break

    train_indices = []
    test_indices = []
    for c in range(10):
        idxs = class_indices[c]
        np.random.shuffle(idxs)
        take = alloc[c]
        train_cls = idxs[:take].tolist()
        test_cls = idxs[take:].tolist()
        train_indices.extend(train_cls)
        test_indices.extend(test_cls)

    # Subsets
    usps_train_subset = Subset(usps_full, train_indices)
    usps_test_subset = Subset(usps_full, test_indices)

    # Wrap with single group id 0
    train_ds = GroupWrapped(usps_train_subset, 0)
    test_ds = GroupWrapped(usps_test_subset, 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)
    return train_loader, test_loader


def build_skewed_mnist_usps_train_subset_loaders(
    root: str,
    batch_size: int,
    num_workers: int,
    mnist_train_size: int,
    usps_train_size: int,
    low_digits: List[int] = None,
    high_digits: List[int] = None,
    mnist_high_ratio: float = 0.7,
    usps_high_ratio: float = 0.3,
    use_full_pool: bool = True,
    seed: int = 1337,
) -> Tuple[DataLoader, DataLoader]:
    """Skewed MNIST+USPS loaders where provided sizes are TRAIN sizes only; TEST is the remainder.

    For MNIST we allocate ``mnist_high_ratio`` of the MNIST training subset to the high digits (``high_digits``)
    and the remainder to low digits (``low_digits``). For USPS we flip the ratios using ``usps_high_ratio``.

    Args:
        root: dataset root
        batch_size: dataloader batch size
        num_workers: dataloader workers
        mnist_train_size: number of MNIST samples to use for training (only)
        usps_train_size: number of USPS samples to use for training (only)
        low_digits: list of low digit classes (default [0,1,2,3,4])
        high_digits: list of high digit classes (default [5,6,7,8,9])
        mnist_high_ratio: fraction of MNIST training subset drawn from high_digits (default 0.7)
        usps_high_ratio: fraction of USPS training subset drawn from high_digits (default 0.3 -> flipped)
        use_full_pool: if True, sample from train+test concatenated pools for broader remainder test coverage
        seed: RNG seed
    Returns:
        train_loader, test_loader
    """
    if low_digits is None:
        low_digits = [0, 1, 2, 3, 4]
    if high_digits is None:
        high_digits = [5, 6, 7, 8, 9]

    assert set(low_digits).isdisjoint(high_digits), "low_digits and high_digits sets must be disjoint"
    assert len(low_digits) + len(high_digits) == 10, "Must cover all 10 digits"

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    def _load_pool(ds_class):
        if not use_full_pool:
            d_train = ds_class(root=root, train=True, transform=transforms.ToTensor(), download=True)
            targets = np.array(d_train.targets)
            return d_train, targets
        d_train = ds_class(root=root, train=True, transform=transforms.ToTensor(), download=True)
        d_test = ds_class(root=root, train=False, transform=transforms.ToTensor(), download=True)
        concat = ConcatDataset([d_train, d_test])
        targets = np.concatenate([np.array(d_train.targets), np.array(d_test.targets)])
        return concat, targets

    mnist_pool, mnist_targets = _load_pool(datasets.MNIST)
    usps_pool, usps_targets = _load_pool(datasets.USPS)

    mnist_class_indices = {c: np.where(mnist_targets == c)[0] for c in range(10)}
    usps_class_indices = {c: np.where(usps_targets == c)[0] for c in range(10)}

    def _alloc(train_size: int, high_ratio: float, low_set: List[int], high_set: List[int], class_indices: dict):
        high_total = int(round(train_size * high_ratio))
        low_total = train_size - high_total
        # Even per-class distribution with remainder handling
        def _split(total: int, classes: List[int]):
            base = total // len(classes)
            rem = total - base * len(classes)
            alloc = {c: base for c in classes}
            # distribute remainder deterministically by class order
            for c in classes[:rem]:
                alloc[c] += 1
            # clip to available counts if scarcity
            for c in classes:
                alloc[c] = min(alloc[c], len(class_indices[c]))
            return alloc
        high_alloc = _split(high_total, high_set)
        low_alloc = _split(low_total, low_set)
        # sample indices without replacement
        chosen = []
        for c, n in {**high_alloc, **low_alloc}.items():
            idxs = class_indices[c].copy()
            rng.shuffle(idxs)
            chosen.extend(idxs[:n].tolist())
        return chosen

    mnist_train_indices = _alloc(mnist_train_size, mnist_high_ratio, low_digits, high_digits, mnist_class_indices)
    usps_train_indices = _alloc(usps_train_size, usps_high_ratio, low_digits, high_digits, usps_class_indices)

    mnist_train_set = Subset(mnist_pool, mnist_train_indices)
    usps_train_set = Subset(usps_pool, usps_train_indices)

    # TEST = remainder of pool (excluding training indices)
    mnist_all = set(range(sum(len(s.dataset) if isinstance(s, Subset) else len(mnist_pool) for s in [mnist_pool]) if isinstance(mnist_pool, ConcatDataset) else len(mnist_pool)))
    usps_all = set(range(sum(len(s.dataset) if isinstance(s, Subset) else len(usps_pool) for s in [usps_pool]) if isinstance(usps_pool, ConcatDataset) else len(usps_pool)))
    mnist_test_indices = list(mnist_all - set(mnist_train_indices))
    usps_test_indices = list(usps_all - set(usps_train_indices))

    mnist_test_set = Subset(mnist_pool, mnist_test_indices)
    usps_test_set = Subset(usps_pool, usps_test_indices)

    # Wrap with group IDs
    train_concat = ConcatDataset([
        GroupWrapped(mnist_train_set, 0),
        GroupWrapped(usps_train_set, 1),
    ])
    test_concat = ConcatDataset([
        GroupWrapped(mnist_test_set, 0),
        GroupWrapped(usps_test_set, 1),
    ])

    train_loader = DataLoader(train_concat, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)
    test_loader = DataLoader(test_concat, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)
    return train_loader, test_loader

