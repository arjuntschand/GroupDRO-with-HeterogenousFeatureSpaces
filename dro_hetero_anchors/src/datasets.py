from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
from torchvision import datasets, transforms

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
    
    # Helper to sample skewed subset
    def sample_skewed(targets, target_size, majority_classes):
        """Sample target_size indices with 80% from majority, 20% from minority."""
        class_indices = {c: np.where(targets == c)[0] for c in range(10)}
        minority_classes = [c for c in range(10) if c not in majority_classes]
        
        num_majority = int(0.8 * target_size)
        num_minority = target_size - num_majority
        
        sampled = []
        # Sample from majority classes uniformly
        for c in majority_classes:
            available = class_indices[c]
            n_per_class = num_majority // len(majority_classes)
            sampled.extend(np.random.choice(available, size=min(n_per_class, len(available)), replace=False))
        
        # Fill remaining majority slots
        remaining = num_majority - len(sampled)
        if remaining > 0:
            all_majority_idxs = np.concatenate([class_indices[c] for c in majority_classes])
            sampled.extend(np.random.choice(all_majority_idxs, size=remaining, replace=True))
        
        # Sample from minority classes uniformly
        for c in minority_classes:
            available = class_indices[c]
            n_per_class = num_minority // len(minority_classes)
            sampled.extend(np.random.choice(available, size=min(n_per_class, len(available)), replace=False))
        
        # Fill remaining minority slots
        remaining = num_minority - (len(sampled) - num_majority)
        if remaining > 0:
            all_minority_idxs = np.concatenate([class_indices[c] for c in minority_classes])
            sampled.extend(np.random.choice(all_minority_idxs, size=remaining, replace=True))
        
        return list(set(sampled))[:target_size]
    
    # Sample skewed MNIST and USPS train indices
    mnist_train_indices = sample_skewed(mnist_targets, mnist_size, mnist_majority)
    usps_train_indices = sample_skewed(usps_targets, usps_size, usps_majority)
    
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
    
    # For test, use the complement (held-out indices from each dataset)
    mnist_all_indices = set(range(len(mnist_train)))
    usps_all_indices = set(range(len(usps_train)))
    
    mnist_test_indices = list(mnist_all_indices - set(mnist_train_indices))
    usps_test_indices = list(usps_all_indices - set(usps_train_indices))
    
    mnist_test_subset = Subset(mnist_train, mnist_test_indices)  # Using mnist_train split as test
    usps_test_subset = Subset(usps_train, usps_test_indices)      # Using usps_train split as test
    
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
    # Ensure we always have USPS test coverage for every class
    usps_reserve_test_frac: float = 0.2,
    usps_min_reserve_per_class: int = 20,
    seed: int = 1337,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/test loaders for 3 groups: MNIST 28x28, USPS 16x16, MNIST 32x32.
    - MNIST samples are drawn from the same base split with disjoint indices between
      the 28x28 and 32x32 groups to avoid overlap.
    - USPS samples are drawn independently from its base split.
    - Each group's train subset is 80/20 skewed toward its configured majority classes.
    - Test subsets are the complements; MNIST complement is split between the two groups
      in proportion to their train sizes to avoid duplicating examples in test.
    """
    if mnist28_majority is None:
        mnist28_majority = [0, 1, 2, 3]
    if usps_majority is None:
        usps_majority = [4, 5, 6]
    if mnist32_majority is None:
        mnist32_majority = [7, 8, 9]

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load base datasets
    mnist_full = datasets.MNIST(root=root, train=True, transform=transforms.ToTensor(), download=True)
    usps_full = datasets.USPS(root=root, train=True, transform=transforms.ToTensor(), download=True)
    mnist_targets = np.array(mnist_full.targets)
    usps_targets = np.array(usps_full.targets)

    # Build per-class pools for MNIST we can draw from without replacement across groups
    mnist_class_pools = {c: np.where(mnist_targets == c)[0].tolist() for c in range(10)}

    def sample_skewed_from_pool(class_pools: dict, target_size: int, majority_classes: List[int]):
        """Sample target_size indices with 80% from majority, 20% from minority, drawing
        without replacement from class_pools. class_pools will be mutated (indices removed)."""
        all_classes = list(range(10))
        minority_classes = [c for c in all_classes if c not in majority_classes]
        num_majority = int(0.8 * target_size)
        num_minority = target_size - num_majority

        sampled: List[int] = []
        # Majority: uniform per-class where possible
        if len(majority_classes) > 0:
            per_cls = max(1, num_majority // len(majority_classes))
            for c in majority_classes:
                take = min(per_cls, len(class_pools[c]))
                if take > 0:
                    choice = np.random.choice(class_pools[c], size=take, replace=False)
                    sampled.extend(choice.tolist())
                    # remove from pool
                    remaining = set(class_pools[c]) - set(choice.tolist())
                    class_pools[c] = list(remaining)
        # Fill remaining majority from combined leftover majority pool (with replacement if needed)
        remaining_maj = num_majority - (len(sampled))
        if remaining_maj > 0:
            combined = sum([class_pools[c] for c in majority_classes], [])
            if len(combined) > 0:
                if remaining_maj <= len(combined):
                    extra = np.random.choice(combined, size=remaining_maj, replace=False)
                    sampled.extend(extra.tolist())
                    for idx in extra.tolist():
                        # remove from the correct pool
                        ci = int(mnist_targets[idx])
                        if idx in class_pools[ci]:
                            class_pools[ci].remove(int(idx))
                else:
                    # not enough to sample without replacement; allow replacement at tail
                    extra = np.random.choice(combined, size=remaining_maj, replace=True)
                    sampled.extend(extra.tolist())

        # Minority: uniform per-class where possible
        if len(minority_classes) > 0:
            per_cls = max(1, num_minority // len(minority_classes))
            for c in minority_classes:
                take = min(per_cls, len(class_pools[c]))
                if take > 0:
                    choice = np.random.choice(class_pools[c], size=take, replace=False)
                    sampled.extend(choice.tolist())
                    remaining = set(class_pools[c]) - set(choice.tolist())
                    class_pools[c] = list(remaining)
        remaining_min = num_minority - (len(sampled) - num_majority)
        if remaining_min > 0:
            combined = sum([class_pools[c] for c in minority_classes], [])
            if len(combined) > 0:
                if remaining_min <= len(combined):
                    extra = np.random.choice(combined, size=remaining_min, replace=False)
                    sampled.extend(extra.tolist())
                    for idx in extra.tolist():
                        ci = int(mnist_targets[idx])
                        if idx in class_pools[ci]:
                            class_pools[ci].remove(int(idx))
                else:
                    extra = np.random.choice(combined, size=remaining_min, replace=True)
                    sampled.extend(extra.tolist())

        # ensure unique and clip size
        sampled = list(dict.fromkeys(sampled))  # preserve order, remove dups
        if len(sampled) > target_size:
            sampled = sampled[:target_size]
        return sampled

    # Sample MNIST group indices without overlap
    mnist28_indices = sample_skewed_from_pool(mnist_class_pools, mnist28_size, mnist28_majority)
    mnist32_indices = sample_skewed_from_pool(mnist_class_pools, mnist32_size, mnist32_majority)

    # Sample USPS indices (independent)
    def sample_skewed_simple(targets, target_size, majority_classes):
        class_indices_all = {c: np.where(targets == c)[0] for c in range(10)}
        # Reserve a subset of each class for test to guarantee coverage
        reserved_for_test: dict[int, List[int]] = {}
        allowed_for_train: dict[int, np.ndarray] = {}
        for c, idxs in class_indices_all.items():
            idxs = np.array(idxs)
            if idxs.size == 0:
                reserved_for_test[c] = []
                allowed_for_train[c] = np.array([], dtype=int)
                continue
            reserve = max(usps_min_reserve_per_class, int(usps_reserve_test_frac * idxs.size))
            reserve = min(reserve, idxs.size)
            keep_for_test = np.random.choice(idxs, size=reserve, replace=False)
            mask = np.ones(idxs.shape[0], dtype=bool)
            mask[np.isin(idxs, keep_for_test)] = False
            allowed = idxs[mask]
            reserved_for_test[c] = keep_for_test.tolist()
            allowed_for_train[c] = allowed
        minority_classes = [c for c in range(10) if c not in majority_classes]
        num_majority = int(0.8 * target_size)
        num_minority = target_size - num_majority
        sampled = []
        # majority
        per_cls = max(1, num_majority // max(1, len(majority_classes)))
        for c in majority_classes:
            available = allowed_for_train[c]
            take = min(per_cls, len(available))
            if take > 0:
                sampled.extend(np.random.choice(available, size=take, replace=False).tolist())
        remaining = num_majority - len(sampled)
        if remaining > 0:
            combined = np.concatenate([allowed_for_train[c] for c in majority_classes]) if len(majority_classes) else np.array([], dtype=int)
            if combined.size > 0:
                # sample without replacement where possible
                replace = remaining > combined.size
                extra = np.random.choice(combined, size=remaining, replace=replace)
                sampled.extend(extra.tolist())
        # minority
        per_cls = max(1, num_minority // max(1, len(minority_classes)))
        for c in minority_classes:
            available = allowed_for_train[c]
            take = min(per_cls, len(available))
            if take > 0:
                sampled.extend(np.random.choice(available, size=take, replace=False).tolist())
        remaining = num_minority - (len(sampled) - num_majority)
        if remaining > 0:
            combined = np.concatenate([allowed_for_train[c] for c in minority_classes]) if len(minority_classes) else np.array([], dtype=int)
            if combined.size > 0:
                replace = remaining > combined.size
                extra = np.random.choice(combined, size=remaining, replace=replace)
                sampled.extend(extra.tolist())
        sampled = list(dict.fromkeys(sampled))
        return sampled[:target_size]

    usps_train_indices = sample_skewed_simple(usps_targets, usps_size, usps_majority)

    # Create train datasets with transforms
    mnist28_tfm = transforms.ToTensor()
    mnist32_tfm = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])  # 28 -> 32 by padding 2 each side
    usps_tfm = transforms.ToTensor()

    mnist_train = datasets.MNIST(root=root, train=True, transform=mnist28_tfm, download=True)
    mnist_train_32 = datasets.MNIST(root=root, train=True, transform=mnist32_tfm, download=True)
    usps_train = datasets.USPS(root=root, train=True, transform=usps_tfm, download=True)

    mnist28_train_subset = Subset(mnist_train, mnist28_indices)
    mnist32_train_subset = Subset(mnist_train_32, mnist32_indices)
    usps_train_subset = Subset(usps_train, usps_train_indices)

    # Wrap groups: 0=mnist28, 1=usps16, 2=mnist32
    train_ds = ConcatDataset([
        GroupWrapped(mnist28_train_subset, 0),
        GroupWrapped(usps_train_subset, 1),
        GroupWrapped(mnist32_train_subset, 2),
    ])

    # Build test sets as complements; split MNIST complement between groups 0 and 2 proportionally
    mnist_all = set(range(len(mnist_train)))
    mnist_used = set(mnist28_indices) | set(mnist32_indices)
    mnist_comp = list(mnist_all - mnist_used)
    np.random.shuffle(mnist_comp)
    if len(mnist_comp) > 0:
        prop_28 = mnist28_size / max(1, (mnist28_size + mnist32_size))
        n28 = int(round(prop_28 * len(mnist_comp)))
        mnist28_test_idx = mnist_comp[:n28]
        mnist32_test_idx = mnist_comp[n28:]
    else:
        mnist28_test_idx, mnist32_test_idx = [], []

    usps_all = set(range(len(usps_train)))
    usps_test_idx = list(usps_all - set(usps_train_indices))

    mnist_test_28 = Subset(datasets.MNIST(root=root, train=True, transform=mnist28_tfm, download=True), mnist28_test_idx)
    mnist_test_32 = Subset(datasets.MNIST(root=root, train=True, transform=mnist32_tfm, download=True), mnist32_test_idx)
    usps_test = Subset(usps_train, usps_test_idx)

    test_ds = ConcatDataset([
        GroupWrapped(mnist_test_28, 0),
        GroupWrapped(usps_test, 1),
        GroupWrapped(mnist_test_32, 2),
    ])

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=pad_to_max_collate)

    return train_loader, test_loader

