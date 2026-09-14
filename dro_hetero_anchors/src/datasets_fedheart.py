"""Fed-Heart Disease dataset loader.

Fed-Heart Disease is a federated binary classification dataset with:
- 4 clients (hospitals) = 4 groups
- 13 tabular features after preprocessing
- Binary classification: heart disease or not

Uses FLamby if installed; otherwise uses a standalone loader that downloads
UCI Heart Disease data and replicates FLamby preprocessing (no pip install flamby needed).
Reference: FLamby - Datasets and Benchmarks for Cross-Silo Federated Learning
"""

from typing import Tuple, List, Optional, Dict
import os
import urllib.request
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split

NUM_CLIENTS = 4  # 4 hospitals: cleveland, hungarian, switzerland, va

# UCI Heart Disease URLs (same as FLamby)
_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
_CENTER_FILES = ["processed.cleveland.data", "processed.hungarian.data", "processed.switzerland.data", "processed.va.data"]


def _download_fedheart_uci(data_dir: str) -> str:
    """Download UCI Heart Disease data to data_dir. Returns data_dir path."""
    os.makedirs(data_dir, exist_ok=True)
    for fname in _CENTER_FILES:
        path = os.path.join(data_dir, fname)
        if os.path.isfile(path):
            continue
        url = _BASE_URL + fname
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {url} to {path}. Error: {e}. "
                "You can manually download the 4 files from "
                "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/ "
                "and place them in the folder: " + data_dir
            ) from e
    return data_dir


# Cache for standalone preprocessed data: (data_dir, seed) -> (features, labels, centers, sets)
_standalone_cache: Dict[tuple, tuple] = {}


def _load_and_preprocess_heart_disease(data_dir: str, train_frac: float = 0.66, seed: int = 43,
                                       impute_missing: bool = False, val_frac: float = 0.0):
    """
    Load UCI Heart Disease data and preprocess like FLamby.
    Returns: features_list, labels_list, centers_list, sets_list (each list of per-sample data),
             center_stats (for normalization). Result is cached by (data_dir, seed).
    """
    # val_frac belongs in the key: a cached split built without a validation set must not be
    # handed back when one is asked for.
    cache_key = (os.path.abspath(data_dir), train_frac, seed, impute_missing, val_frac)
    if cache_key in _standalone_cache:
        return _standalone_cache[cache_key]
    data_dir = _download_fedheart_uci(data_dir)
    centers_number = {"cleveland": 0, "hungarian": 1, "switzerland": 2, "va": 3}

    all_features = []
    all_labels = []
    all_centers = []
    all_sets = []
    center_dfs = []  # (center_id, train_df, test_df) for later normalization

    for fname in _CENTER_FILES:
        center_name = fname.replace("processed.", "").replace(".data", "")
        cid = centers_number[center_name]
        path = os.path.join(data_dir, fname)

        df = pd.read_csv(path, header=None)
        df = df.replace("?", np.nan).drop([10, 11, 12], axis=1)
        df = df.apply(pd.to_numeric, errors="coerce")
        if impute_missing:
            # Keep every patient and fill missing values with that SITE's median (the label
            # column is never missing, so it is untouched). Dropping rows instead discards
            # 63% of Switzerland, which is missing cholesterol on most records, leaving only
            # ~10 test samples for that group. FLamby's own benchmark imputes rather than drops.
            feat = df.columns[:-1]
            df[feat] = df[feat].fillna(df[feat].median())
            df = df.dropna(axis=0)          # drops only rows with a missing LABEL
        else:
            df = df.dropna(axis=0)

        center_X = df.iloc[:, :-1]
        center_y = df.iloc[:, -1]

        nb = int(len(center_X))
        current_labels = center_y.where(center_y == 0, 1, inplace=False)
        levels = np.unique(current_labels)
        if len(np.unique(current_labels)) > 1 and all((current_labels == lev).sum() > 2 for lev in levels):
            stratify = current_labels
        else:
            stratify = None
        indices_train, indices_test = train_test_split(
            np.arange(nb),
            test_size=1.0 - train_frac,
            train_size=train_frac,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )

        # Carve the validation set out of TRAIN, per centre, so every group is represented in
        # it. The test fold is never touched.
        indices_val = np.array([], dtype=int)
        if val_frac and val_frac > 0 and len(indices_train) > 4:
            strat_tr = current_labels.values[indices_train]
            try:
                indices_train, indices_val = train_test_split(
                    indices_train, test_size=val_frac, random_state=seed, shuffle=True,
                    stratify=strat_tr)
            except ValueError:      # a class too small to stratify at this centre
                indices_train, indices_val = train_test_split(
                    indices_train, test_size=val_frac, random_state=seed, shuffle=True)
        train_set, val_set = set(indices_train.tolist()), set(indices_val.tolist())

        for i in range(nb):
            x_row = center_X.iloc[i : i + 1]
            y_val = center_y.iloc[i]
            all_centers.append(cid)
            all_sets.append("train" if i in train_set else ("val" if i in val_set else "test"))
            all_features.append(x_row)
            all_labels.append(y_val)

        train_idx = [i for i in range(nb) if i in train_set]
        test_idx = [i for i in range(nb) if i in set(indices_test.tolist())]
        center_dfs.append((cid, center_X.iloc[train_idx], center_X.iloc[test_idx], center_y.iloc[train_idx], center_y.iloc[test_idx]))

    # Concatenate and one-hot encode (get_dummies on columns 2 and 6)
    features_df = pd.concat(all_features, ignore_index=True)
    features_df = pd.get_dummies(features_df, columns=[2, 6], drop_first=True)
    assert features_df.shape[1] == 13, f"Expected 13 features, got {features_df.shape[1]}"

    labels = np.array(all_labels, dtype=np.float32)
    labels = np.where(labels == 0, 0, 1)  # 0 = no disease, 1 = disease

    features_list = [torch.from_numpy(features_df.iloc[i].values.astype(np.float32)) for i in range(len(features_df))]

    # Per-center normalization (using train stats per center)
    center_stats = {}
    for cid in range(NUM_CLIENTS):
        mask = [i for i in range(len(all_centers)) if all_centers[i] == cid and all_sets[i] == "train"]
        if not mask:
            center_stats[cid] = {"mean": torch.zeros(13), "std": torch.ones(13)}
            continue
        tensors = [features_list[i] for i in mask]
        stack = torch.stack(tensors, dim=0)
        center_stats[cid] = {"mean": stack.mean(dim=0), "std": stack.std(dim=0) + 1e-9}

    # Normalize each sample by its center's train stats
    normalized = []
    for i in range(len(features_list)):
        cid = all_centers[i]
        x = (features_list[i] - center_stats[cid]["mean"]) / center_stats[cid]["std"]
        normalized.append(x)

    out = (normalized, labels, all_centers, all_sets, center_stats)
    _standalone_cache[cache_key] = out
    return out


class _StandaloneHeartDiseaseDataset(Dataset):
    """Standalone UCI-based dataset (no FLamby). One center, train or test."""

    def __init__(self, center: int, train: bool, data_dir: str, seed: int = 43, train_frac: float = 0.66,
                 impute_missing: bool = False, val_frac: float = 0.0, split: Optional[str] = None):
        self.center = center
        self.train = train
        # `train` stays for the existing call sites; `split` is what actually selects, so "val"
        # can be requested without changing any of them.
        want = split or ("train" if train else "test")
        features, labels, centers, sets, _ = _load_and_preprocess_heart_disease(
            data_dir, train_frac=train_frac, seed=seed, impute_missing=impute_missing,
            val_frac=val_frac)
        self.features = [f for i, f in enumerate(features) if centers[i] == center and sets[i] == want]
        self.labels = [labels[i] for i in range(len(centers)) if centers[i] == center and sets[i] == want]
        self.labels = torch.from_numpy(np.array(self.labels, dtype=np.int64))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].clone()
        y = self.labels[idx]
        return x, y, self.center


def _use_flamby() -> bool:
    try:
        from flamby.datasets.fed_heart_disease import FedHeartDisease  # noqa: F401
        return True
    except ImportError:
        return False


class FedHeartDiseaseDataset(Dataset):
    """Wrapper for Fed-Heart Disease that returns (x, y, g) tuples.
    Uses FLamby if available; otherwise use CombinedFedHeartDataset with data_root (standalone).
    """
    
    def __init__(self, center: int = 0, train: bool = True, pooled: bool = False, data_dir: Optional[str] = None):
        if _use_flamby():
            from flamby.datasets.fed_heart_disease import FedHeartDisease
            self.center = center
            self.train = train
            self.pooled = pooled
            if pooled:
                self.dataset = FedHeartDisease(train=train, pooled=True)
                self.group_id = -1
                self._build_center_mapping_flamby(train)
            else:
                self.dataset = FedHeartDisease(center=center, train=train, pooled=False)
                self.group_id = center
            self._standalone = False
            return
        if data_dir is None:
            raise ImportError(
                "FLamby is not installed and no data_dir was provided.\n"
                "Either: pip install git+https://github.com/owkin/FLamby.git\n"
                "Or: use build_fedheart_loaders(..., data_root='datasets/fed_heart_disease') "
                "to auto-download UCI data (no FLamby needed)."
            )
        self._standalone = True
        self.dataset = _StandaloneHeartDiseaseDataset(center=center, train=train, data_dir=data_dir)
        self.group_id = center
        self.pooled = False
    
    def _build_center_mapping_flamby(self, train: bool):
        from flamby.datasets.fed_heart_disease import FedHeartDisease
        self.center_mapping = []
        for c in range(NUM_CLIENTS):
            ds = FedHeartDisease(center=c, train=train, pooled=False)
            self.center_mapping.extend([c] * len(ds))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self._standalone:
            x, y, g = self.dataset[idx]
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.long)
            return x.float(), y.view(-1).squeeze(0), g
        x, y = self.dataset[idx]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        if isinstance(y, torch.Tensor):
            y = y.long().squeeze()
        else:
            y = torch.tensor(y, dtype=torch.long)
        g = self.center_mapping[idx] if self.pooled else self.group_id
        return x, y, g


class CombinedFedHeartDataset(Dataset):
    """Combines all Fed-Heart Disease centers into one dataset with group labels.
    Uses FLamby if installed; otherwise uses standalone UCI download when data_root is set.
    
    Supports several techniques to create challenging scenarios for GroupDRO experiments:
    - group_max_samples: Subsample training so groups are imbalanced
    - label_noise_rate: Per-group label noise rates (flip labels to create difficulty)
    - feature_mask: Per-group feature masks (simulate heterogeneous feature spaces)
    - input_noise_std: Per-group input noise (make certain groups harder)
    """
    
    def __init__(self, train: bool = True, centers: Optional[List[int]] = None, data_root: Optional[str] = None,
                 group_max_samples: Optional[List[Optional[int]]] = None, seed: int = 43,
                 train_frac: float = 0.66,
                 label_noise_rate: Optional[List[float]] = None,
                 feature_mask: Optional[List[Optional[List[int]]]] = None,
                 input_noise_std: Optional[List[float]] = None,
                 subsample_seed: Optional[int] = None,
                 impute_missing: bool = False,
                 val_frac: float = 0.0,
                 split: Optional[str] = None):
        self.train = train
        self.split = split or ("train" if train else "test")
        self.centers = centers if centers is not None else list(range(NUM_CLIENTS))
        self.seed = seed
        self.train_frac = train_frac

        # Store noise/masking parameters for __getitem__
        self.label_noise_rate = label_noise_rate  # e.g. [0.3, 0.3, 0.0, 0.0] = 30% noise on G0,G1
        self.feature_mask = feature_mask  # e.g. [None, None, [0,1,2], [0,1,2]] = G2,G3 only see features 0,1,2
        self.input_noise_std = input_noise_std  # e.g. [0.0, 0.0, 0.5, 0.5] = add noise to G2,G3

        # Precompute label flips for reproducibility
        self._label_flips = {}  # (group, local_idx) -> flipped_label
        self._rng = np.random.default_rng(seed + 12345)  # separate RNG for noise

        if _use_flamby():
            self._standalone = False
            self.datasets = [
                FedHeartDiseaseDataset(center=c, train=train, pooled=False)
                for c in self.centers
            ]
        else:
            self._standalone = True
            if data_root is None:
                data_root = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "fed_heart_disease")
            data_root = os.path.abspath(data_root)
            self.datasets = [
                _StandaloneHeartDiseaseDataset(center=c, train=train, data_dir=data_root, seed=seed,
                                               train_frac=train_frac, impute_missing=impute_missing,
                                               val_frac=val_frac, split=self.split)
                for c in self.centers
            ]

        # Optional per-group subsampling (for paper: create imbalance so ERM underperforms on minority group)
        # subsample_seed allows varying which samples are capped while keeping train/test split fixed
        rng = np.random.default_rng(subsample_seed if subsample_seed is not None else seed)
        self.indices = []  # list of (group_id, local_idx)
        for g, ds in enumerate(self.datasets):
            n = len(ds)
            if group_max_samples is not None and g < len(group_max_samples) and group_max_samples[g] is not None:
                cap = min(n, group_max_samples[g])
                idxs = rng.choice(n, size=cap, replace=False)
                for i in idxs:
                    self.indices.append((g, int(i)))
            else:
                for i in range(n):
                    self.indices.append((g, i))
        
        if len(self.indices) == 0:
            raise ValueError("Subsampling left 0 samples. Check group_max_samples.")
        
        self.cumulative_sizes = [0]
        self.group_ids = list(self.centers)
        for ds in self.datasets:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(ds))
        self._full_size = self.cumulative_sizes[-1]
        self.total_size = len(self.indices) if self.indices else self._full_size
        
        # Precompute label flips for training set (reproducible)
        if train and self.label_noise_rate is not None:
            for g, local_idx in self.indices:
                if g < len(self.label_noise_rate) and self.label_noise_rate[g] > 0:
                    if self._rng.random() < self.label_noise_rate[g]:
                        # Get original label and flip it
                        _, orig_y, _ = self.datasets[g][local_idx]
                        orig_label = int(orig_y.item()) if hasattr(orig_y, 'item') else int(orig_y)
                        self._label_flips[(g, local_idx)] = 1 - orig_label  # flip binary label
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if self.indices:
            g, local_idx = self.indices[idx]
            x, y, _ = self.datasets[g][local_idx]
        else:
            # no subsampling: use cumulative ranges
            for i, (start, end) in enumerate(zip(self.cumulative_sizes[:-1], self.cumulative_sizes[1:])):
                if start <= idx < end:
                    local_idx = idx - start
                    x, y, _ = self.datasets[i][local_idx]
                    g = self.group_ids[i]
                    break
            else:
                raise IndexError(f"Index {idx} out of range for dataset of size {self.total_size}")
        
        # Apply label noise (training only, precomputed for reproducibility)
        if self.train and (g, local_idx) in self._label_flips:
            y = torch.tensor(self._label_flips[(g, local_idx)], dtype=torch.long)
        
        # Apply feature masking (simulate heterogeneous feature spaces)
        if self.feature_mask is not None and g < len(self.feature_mask) and self.feature_mask[g] is not None:
            mask_indices = self.feature_mask[g]
            # Zero out features NOT in the mask
            x_masked = torch.zeros_like(x)
            for feat_idx in mask_indices:
                if feat_idx < x.shape[0]:
                    x_masked[feat_idx] = x[feat_idx]
            x = x_masked
        
        # Apply input noise (make certain groups harder)
        if self.train and self.input_noise_std is not None and g < len(self.input_noise_std):
            noise_std = self.input_noise_std[g]
            if noise_std > 0:
                x = x + torch.randn_like(x) * noise_std
        
        return x, y, g
    
    def get_group_counts(self) -> List[int]:
        """Return the number of samples per group (after subsampling if used)."""
        num_g = len(self.datasets)
        counts = [0] * num_g
        if self.indices:
            for g, _ in self.indices:
                counts[g] += 1
        else:
            for i, ds in enumerate(self.datasets):
                counts[i] = len(ds)
        return counts
    
    def get_class_counts(self) -> List[int]:
        """Return the number of samples per class (across all groups, after subsampling if used)."""
        counts = [0, 0]
        if self.indices:
            for g, local_idx in self.indices:
                _, y, _ = self.datasets[g][local_idx]
                counts[int(y.item())] += 1
        else:
            for ds in self.datasets:
                for i in range(len(ds)):
                    _, y, _ = ds[i]
                    counts[int(y.item())] += 1
        return counts
    
    def get_group_class_counts(self) -> List[List[int]]:
        """Return per-group-per-class counts (after subsampling if used)."""
        num_g = len(self.datasets)
        counts = [[0, 0] for _ in range(num_g)]
        if self.indices:
            for g, local_idx in self.indices:
                _, y, _ = self.datasets[g][local_idx]
                counts[g][int(y.item())] += 1
        else:
            for g_idx, ds in enumerate(self.datasets):
                for i in range(len(ds)):
                    _, y, _ = ds[i]
                    counts[g_idx][int(y.item())] += 1
        return counts


class StratifiedGroupSampler(Sampler):
    """Sampler that maintains group proportions π in each batch.
    
    This ensures each batch has approximately the same group distribution
    as the full dataset, which is critical for stable GroupDRO training.
    
    Args:
        dataset: Dataset with group labels (must have (x, y, g) format)
        batch_size: Desired batch size
        group_counts: List of sample counts per group
        shuffle: Whether to shuffle within groups
        drop_last: Whether to drop the last incomplete batch
    """
    
    def __init__(self, dataset: Dataset, batch_size: int, group_counts: List[int],
                 shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_counts = group_counts
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.num_groups = len(group_counts)
        self.total_samples = sum(group_counts)
        
        # Compute group proportions π
        self.pi = [c / self.total_samples for c in group_counts]
        
        # Compute samples per group per batch
        self.samples_per_group = []
        remaining = batch_size
        for g in range(self.num_groups - 1):
            n = max(1, int(round(self.pi[g] * batch_size)))
            n = min(n, remaining - (self.num_groups - 1 - g))  # Ensure at least 1 per remaining group
            self.samples_per_group.append(n)
            remaining -= n
        self.samples_per_group.append(remaining)  # Last group gets the remainder
        
        # Build index lists per group
        self._build_group_indices()
    
    def _build_group_indices(self):
        """Build lists of indices for each group."""
        self.group_indices = [[] for _ in range(self.num_groups)]
        
        for idx in range(len(self.dataset)):
            _, _, g = self.dataset[idx]
            if isinstance(g, torch.Tensor):
                g = g.item()
            self.group_indices[g].append(idx)
    
    def __iter__(self):
        # Shuffle indices within each group if requested
        group_indices = []
        for g in range(self.num_groups):
            indices = self.group_indices[g].copy()
            if self.shuffle:
                np.random.shuffle(indices)
            group_indices.append(indices)
        
        # Track position in each group
        group_pos = [0] * self.num_groups
        
        # Generate batches
        batches = []
        while True:
            batch = []
            can_continue = True
            
            for g in range(self.num_groups):
                n_samples = self.samples_per_group[g]
                start = group_pos[g]
                end = start + n_samples
                
                # If not enough samples left in this group, stop (one epoch, no wrap)
                if end > len(group_indices[g]):
                    can_continue = False
                    break
                
                batch.extend(group_indices[g][start:end])
                group_pos[g] = end
            
            if not can_continue:
                if len(batch) > 0 and not self.drop_last:
                    batches.append(batch)
                break
            
            if len(batch) == self.batch_size:
                batches.append(batch)
            elif not self.drop_last and len(batch) > 0:
                batches.append(batch)
            
            all_exhausted = all(group_pos[g] >= len(group_indices[g]) for g in range(self.num_groups))
            if all_exhausted:
                break
        
        # Shuffle batch order
        if self.shuffle:
            np.random.shuffle(batches)
        
        for batch in batches:
            yield from batch
    
    def __len__(self):
        if self.drop_last:
            return (self.total_samples // self.batch_size) * self.batch_size
        else:
            return self.total_samples


def collate_fedheart(batch):
    """Collate function for Fed-Heart Disease batches.
    
    Args:
        batch: List of (x, y, g) tuples
        
    Returns:
        x: Tensor of shape (batch_size, 13)
        y: Tensor of shape (batch_size,)
        g: Tensor of shape (batch_size,)
    """
    xs, ys, gs = zip(*batch)
    
    x = torch.stack(xs, dim=0)
    y = torch.stack([yi if isinstance(yi, torch.Tensor) else torch.tensor(yi) for yi in ys], dim=0)
    g = torch.tensor(gs, dtype=torch.long)
    
    return x, y, g


def build_fedheart_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 1337,
    stratified: bool = True,
    centers: Optional[List[int]] = None,
    data_root: Optional[str] = None,
    group_max_train_samples: Optional[List[Optional[int]]] = None,
    train_frac: float = 0.66,
    label_noise_rate: Optional[List[float]] = None,
    feature_mask: Optional[List[Optional[List[int]]]] = None,
    input_noise_std: Optional[List[float]] = None,
    subsample_seed: Optional[int] = None,
    impute_missing: bool = False,
    val_frac: float = 0.0,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Build train and test DataLoaders for Fed-Heart Disease.
    
    If the FLamby package is not installed, uses a standalone loader that downloads
    UCI Heart Disease data to data_root (default: repo datasets/fed_heart_disease).
    
    Args:
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        seed: Random seed
        stratified: If True, use stratified sampling to maintain group proportions
        centers: List of center indices to include (default: all 4)
        data_root: Directory for UCI data when not using FLamby (default: datasets/fed_heart_disease)
        group_max_train_samples: If set, cap training samples per group (e.g. [None, None, 25, None]
            to make group 2 a 25-sample minority so ERM underperforms and GroupDRO can show strong gain).
        train_frac: Fraction of data used for training (default 0.66). Use 0.8 for standard 80/20 split.
        label_noise_rate: Per-group label noise rates (e.g. [0.3, 0.3, 0.0, 0.0] = 30% noise on G0,G1).
            ERM will fit the noise; GroupDRO will upweight clean minority groups.
        feature_mask: Per-group feature indices to keep (e.g. [[0,1,2], [3,4,5], None, None]).
            Simulates heterogeneous feature spaces where groups have different available features.
        input_noise_std: Per-group input noise std (e.g. [0.0, 0.0, 0.5, 0.5]).
            Makes certain groups harder by adding Gaussian noise to inputs.
        
    Returns:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        dataset_info: Dictionary with dataset statistics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if data_root is None and not _use_flamby():
        data_root = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "fed_heart_disease")
        data_root = os.path.abspath(data_root)
    
    # Create datasets (subsample train only when group_max_train_samples is set)
    train_dataset = CombinedFedHeartDataset(
        train=True, centers=centers, data_root=data_root,
        group_max_samples=group_max_train_samples, seed=seed,
        train_frac=train_frac, val_frac=val_frac,
        label_noise_rate=label_noise_rate,
        feature_mask=feature_mask,
        input_noise_std=input_noise_std,
        subsample_seed=subsample_seed,
        impute_missing=impute_missing,
    )
    test_dataset = CombinedFedHeartDataset(
        train=False, centers=centers, data_root=data_root,
        seed=seed, train_frac=train_frac, val_frac=val_frac,
        impute_missing=impute_missing,
    )
    # Held out from train, used only to pick the reported epoch. Feature masking and noise are
    # applied exactly as on train so the two are measured on the same input distribution.
    val_dataset = (CombinedFedHeartDataset(
        train=False, centers=centers, data_root=data_root, split="val",
        seed=seed, train_frac=train_frac, val_frac=val_frac,
        label_noise_rate=None, feature_mask=feature_mask,
        input_noise_std=input_noise_std, impute_missing=impute_missing,
    ) if val_frac and val_frac > 0 else None)
    
    # Get statistics
    train_group_counts = train_dataset.get_group_counts()
    test_group_counts = test_dataset.get_group_counts()
    train_class_counts = train_dataset.get_class_counts()
    test_class_counts = test_dataset.get_class_counts()
    train_group_class_counts = train_dataset.get_group_class_counts()
    test_group_class_counts = test_dataset.get_group_class_counts()
    
    dataset_info = {
        "num_groups": len(train_group_counts),
        "num_classes": 2,
        "train_total": sum(train_group_counts),
        "test_total": sum(test_group_counts),
        "train_group_counts": train_group_counts,
        "test_group_counts": test_group_counts,
        "train_class_counts": train_class_counts,
        "test_class_counts": test_class_counts,
        "train_group_class_counts": train_group_class_counts,
        "test_group_class_counts": test_group_class_counts,
        "group_proportions": [c / sum(train_group_counts) for c in train_group_counts],
    }
    
    # Create samplers
    if stratified:
        train_sampler = StratifiedGroupSampler(
            train_dataset, batch_size, train_group_counts, shuffle=True, drop_last=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fedheart,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fedheart,
            pin_memory=True,
        )
    
    # Test loader doesn't need stratified sampling
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fedheart,
        pin_memory=True,
    )

    if val_dataset is not None and len(val_dataset) > 0:
        dataset_info["val_loader"] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=collate_fedheart, pin_memory=True)
        dataset_info["val_group_counts"] = val_dataset.get_group_counts()
        dataset_info["val_total"] = len(val_dataset)

    return train_loader, test_loader, dataset_info


def print_fedheart_summary(dataset_info: Dict):
    """Pretty-print Fed-Heart Disease dataset statistics."""
    print("=" * 60)
    print("Fed-Heart Disease Dataset Summary")
    print("=" * 60)
    print(f"Number of groups (hospitals): {dataset_info['num_groups']}")
    print(f"Number of classes: {dataset_info['num_classes']} (binary: no disease / disease)")
    print()
    print(f"Training samples: {dataset_info['train_total']}")
    print(f"Test samples: {dataset_info['test_total']}")
    print()
    print("Per-group distribution (training):")
    for g, count in enumerate(dataset_info['train_group_counts']):
        pct = 100 * count / dataset_info['train_total']
        print(f"  Group {g} (Hospital {g}): {count:4d} samples ({pct:5.1f}%)")
    print()
    print("Per-class distribution (training):")
    for c, count in enumerate(dataset_info['train_class_counts']):
        pct = 100 * count / dataset_info['train_total']
        label = "No Disease" if c == 0 else "Disease"
        print(f"  Class {c} ({label}): {count:4d} samples ({pct:5.1f}%)")
    print()
    print("Per-group-per-class breakdown (training):")
    for g, class_counts in enumerate(dataset_info['train_group_class_counts']):
        print(f"  Group {g}: Class 0={class_counts[0]}, Class 1={class_counts[1]}")
    print()
    print(f"Group proportions π: {[f'{p:.3f}' for p in dataset_info['group_proportions']]}")
    print("=" * 60)


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing Fed-Heart Disease dataset loader...")
    
    train_loader, test_loader, info = build_fedheart_loaders(
        batch_size=32, stratified=True
    )
    
    print_fedheart_summary(info)
    
    # Test a few batches
    print("\nTesting batches:")
    for i, (x, y, g) in enumerate(train_loader):
        if i >= 3:
            break
        print(f"Batch {i}: x.shape={x.shape}, y.shape={y.shape}, g.shape={g.shape}")
        print(f"  Group distribution: {[int((g == gid).sum()) for gid in range(info['num_groups'])]}")
        print(f"  Class distribution: {[int((y == cid).sum()) for cid in range(2)]}")
