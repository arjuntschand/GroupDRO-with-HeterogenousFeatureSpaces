"""
TextCaps Dataset Loader for GroupDRO with Heterogeneous Feature Spaces

Multi-modal setup:
- Group 0: Visual features (CNN encoder on images)
- Group 1: Text features (text encoder on OCR/caption text)
- Group 2 (optional): Combined visual+text features (concatenated latents)

Each image can appear 2 or 3 times in the dataset (once per modality).
This tests whether GroupDRO can align visual and textual representations
through the shared anchor space.

Supports two data sources:
1. Hugging Face datasets (recommended) - `lmms-lab/TextCaps`
2. Local JSON annotations + downloaded images (legacy)
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
from torchvision import transforms
from PIL import Image

# Try to import huggingface datasets
try:
    from datasets import load_dataset, concatenate_datasets
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Stratified Batch Sampler
# ---------------------------------------------------------------------------

class StratifiedGroupSampler(Sampler):
    """Sampler that maintains group distribution π within each batch.
    
    Ensures batches reflect the overall dataset group distribution, which is
    important for consistent GroupDRO training with π-proportional weighting.
    """
    
    def __init__(self, group_labels: List[int], batch_size: int, shuffle: bool = True, seed: int = 1337):
        """
        Args:
            group_labels: List of group IDs for each sample in the dataset
            batch_size: Desired batch size
            shuffle: Whether to shuffle within groups
            seed: Random seed for reproducibility
        """
        self.group_labels = np.array(group_labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        
        # Get unique groups and their indices
        self.groups = np.unique(self.group_labels)
        self.group_indices = {g: np.where(self.group_labels == g)[0] for g in self.groups}
        
        # Compute group proportions (π)
        total = len(group_labels)
        self.group_proportions = {g: len(self.group_indices[g]) / total for g in self.groups}
        
        # Compute samples per group per batch (rounded, at least 1)
        self.samples_per_group = {}
        for g in self.groups:
            n = max(1, int(round(self.group_proportions[g] * batch_size)))
            self.samples_per_group[g] = n
        
        # Adjust to match batch_size exactly
        total_per_batch = sum(self.samples_per_group.values())
        if total_per_batch != batch_size:
            # Adjust the largest group
            largest_group = max(self.groups, key=lambda g: self.samples_per_group[g])
            diff = batch_size - total_per_batch
            self.samples_per_group[largest_group] = max(1, self.samples_per_group[largest_group] + diff)
    
    def __iter__(self):
        # Shuffle indices within each group if needed
        group_iters = {}
        for g in self.groups:
            indices = self.group_indices[g].copy()
            if self.shuffle:
                self.rng.shuffle(indices)
            group_iters[g] = iter(indices)
        
        # Generate batches
        batches = []
        exhausted = set()
        
        while len(exhausted) < len(self.groups):
            batch = []
            for g in self.groups:
                if g in exhausted:
                    continue
                n = self.samples_per_group[g]
                for _ in range(n):
                    try:
                        idx = next(group_iters[g])
                        batch.append(idx)
                    except StopIteration:
                        exhausted.add(g)
                        break
            
            if len(batch) > 0:
                if self.shuffle:
                    self.rng.shuffle(batch)
                batches.append(batch)
        
        # Flatten batches into a single list of indices
        for batch in batches:
            yield from batch
    
    def __len__(self):
        return len(self.group_labels)


# ---------------------------------------------------------------------------
# Text Tokenizer (simple bag-of-words / character-level for now)
# ---------------------------------------------------------------------------

class SimpleTextEncoder:
    """Simple text encoder that converts text to a fixed-size vector.
    
    Uses character-level encoding with a small vocabulary for simplicity.
    Can be replaced with BERT embeddings later.
    """
    
    def __init__(self, max_len: int = 128, embed_dim: int = 64):
        self.max_len = max_len
        self.embed_dim = embed_dim
        # Simple character vocabulary (lowercase letters, digits, space, punctuation)
        self.vocab = {c: i+1 for i, c in enumerate(
            'abcdefghijklmnopqrstuvwxyz0123456789 .,!?-\'\"'
        )}
        self.vocab_size = len(self.vocab) + 1  # +1 for padding/unknown
        
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to a sequence of token IDs."""
        text = text.lower()
        tokens = [self.vocab.get(c, 0) for c in text[:self.max_len]]
        # Pad to max_len
        tokens = tokens + [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset Classes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Hugging Face Dataset Classes (Recommended)
# ---------------------------------------------------------------------------

class TextCapsHFDataset(Dataset):
    """TextCaps dataset loaded from Hugging Face.
    
    Each sample appears 2 or 3 times depending on include_combined:
    - Group 0: visual only
    - Group 1: text only
    - Group 2 (optional): combined visual+text
    
    This is much more reliable than downloading from Flickr URLs.
    
    Uses LAZY LOADING - images are loaded on-demand in __getitem__, not stored in memory.
    """
    
    def __init__(
        self,
        hf_dataset,
        class_to_idx: Dict[str, int],
        text_encoder,
        transform=None,
        max_samples: Optional[int] = None,
        include_combined: bool = False,
        spurious_correlation: bool = False,
        spurious_majority_frac: float = 0.8,
        spurious_group_classes: Optional[Dict[int, List[int]]] = None,
        seed: int = 1337,
    ):
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.text_encoder = text_encoder
        self.class_to_idx = class_to_idx
        self.include_combined = include_combined
        self.spurious_correlation = spurious_correlation
        
        # Store reference to HF dataset for lazy loading
        self.hf_dataset = hf_dataset
        
        rng = np.random.RandomState(seed)
        
        # Build samples list
        # IMPORTANT: Don't store PIL images! Store indices for lazy loading.
        self.samples = []
        count = 0
        skipped = 0
        
        # Default spurious correlation: group 0 -> class 0, group 1 -> classes 1,2, group 2 -> class 3
        # For 4-class (Bottle=0, Car=1, Food=2, Book=3):
        #   Group 0 (visual): 80% Bottle
        #   Group 1 (text): 80% Car+Food (40% each)
        #   Group 2 (combined): 80% Book
        if spurious_group_classes is None and spurious_correlation:
            spurious_group_classes = {
                0: [0],        # Group 0 majority: Bottle
                1: [1, 2],     # Group 1 majority: Car, Food
                2: [3],        # Group 2 majority: Book
            }
        
        for hf_idx, item in enumerate(hf_dataset):
            # Get primary class
            image_classes = item.get('image_classes', [])
            if image_classes and image_classes[0] in class_to_idx:
                label = class_to_idx[image_classes[0]]
                
                # caption_str is a list in HF dataset - use first caption
                caption_field = item.get('caption_str', item.get('reference_strs', ['']))
                if isinstance(caption_field, list):
                    caption = caption_field[0] if caption_field else ''
                else:
                    caption = str(caption_field)
                
                if spurious_correlation:
                    # Spurious correlation mode: assign image to ONE group based on class
                    # Determine which group this class is "majority" in
                    majority_group = None
                    for g, classes in spurious_group_classes.items():
                        if label in classes:
                            majority_group = g
                            break
                    
                    # With prob=majority_frac, assign to majority group; else random other group
                    num_groups = 3 if include_combined else 2
                    if majority_group is not None and rng.random() < spurious_majority_frac:
                        assigned_group = majority_group
                    else:
                        # Assign to a random group (could be any, including minority)
                        other_groups = [g for g in range(num_groups) if g != majority_group]
                        if other_groups:
                            assigned_group = rng.choice(other_groups)
                        else:
                            assigned_group = rng.randint(0, num_groups)
                    
                    # Create sample for the assigned group only
                    modality = ['visual', 'text', 'combined'][assigned_group]
                    self.samples.append({
                        'modality': modality,
                        'hf_idx': hf_idx,
                        'image_id': item.get('image_id', str(count)),
                        'caption': caption,
                        'label': label,
                        'group': assigned_group,
                    })
                else:
                    # Standard mode: create samples for all groups
                    # Visual sample (group 0) - store HF index, NOT the image!
                    self.samples.append({
                        'modality': 'visual',
                        'hf_idx': hf_idx,  # Store index for lazy loading
                        'image_id': item.get('image_id', str(count)),
                        'caption': caption,  # Store caption for combined mode
                        'label': label,
                        'group': 0,
                    })
                    
                    # Text sample (group 1)
                    self.samples.append({
                        'modality': 'text',
                        'hf_idx': hf_idx,  # Also store for combined lookup
                        'caption': caption,
                        'image_id': item.get('image_id', str(count)),
                        'label': label,
                        'group': 1,
                    })
                    
                    # Combined sample (group 2) - only if enabled
                    if include_combined:
                        self.samples.append({
                            'modality': 'combined',
                            'hf_idx': hf_idx,
                            'caption': caption,
                            'image_id': item.get('image_id', str(count)),
                            'label': label,
                            'group': 2,
                        })
                
                count += 1
                if max_samples and count >= max_samples:
                    break
            else:
                skipped += 1
        
        if skipped > 0:
            print(f"Skipped {skipped} samples (no valid class or class not in mapping)")
        
        if spurious_correlation:
            print(f"Created {len(self.samples)} samples from {count} images (spurious correlation mode, majority_frac={spurious_majority_frac})")
        else:
            samples_per_image = 3 if include_combined else 2
            print(f"Created {len(self.samples)} samples from {count} images ({samples_per_image}x per image, lazy loading enabled)")
        
        # Store group counts for stratified sampling and π initialization
        self.group_counts = self._compute_group_counts()
    
    def _compute_group_counts(self) -> Dict[int, int]:
        """Count samples per group."""
        counts = {}
        for sample in self.samples:
            g = sample['group']
            counts[g] = counts.get(g, 0) + 1
        return counts
    
    def get_class_distribution_per_group(self) -> Dict[int, Dict[int, int]]:
        """Get class distribution within each group.
        
        Returns:
            Dict mapping group_id -> {class_id -> count}
        """
        dist = {}
        for sample in self.samples:
            g = sample['group']
            c = sample['label']
            if g not in dist:
                dist[g] = {}
            dist[g][c] = dist[g].get(c, 0) + 1
        return dist
    
    def get_group_labels(self) -> List[int]:
        """Return list of group IDs for stratified sampling."""
        return [s['group'] for s in self.samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        if item['modality'] == 'visual':
            try:
                # Lazy load image from HF dataset
                hf_item = self.hf_dataset[item['hf_idx']]
                img = hf_item['image'].convert('RGB')
                x = self.transform(img)
            except Exception:
                x = torch.zeros(3, 224, 224)
            return x, item['label'], item['group']
        
        elif item['modality'] == 'text':
            x = self.text_encoder.encode(item['caption'])
            return x, item['label'], item['group']
        
        else:  # combined
            # Return both visual and text for combined processing
            try:
                hf_item = self.hf_dataset[item['hf_idx']]
                img = hf_item['image'].convert('RGB')
                x_visual = self.transform(img)
            except Exception:
                x_visual = torch.zeros(3, 224, 224)
            
            x_text = self.text_encoder.encode(item['caption'])
            
            # Return as tuple (visual, text) for combined processing
            return (x_visual, x_text), item['label'], item['group']


# ---------------------------------------------------------------------------
# Local File Dataset Classes (Legacy)
# ---------------------------------------------------------------------------

class TextCapsVisualDataset(Dataset):
    """TextCaps dataset - Visual modality (Group 0).
    
    Returns image tensors with class labels.
    """
    
    def __init__(
        self,
        annotations_path: str,
        images_dir: str,
        class_to_idx: Dict[str, int],
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.class_to_idx = class_to_idx
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        # Filter to samples with valid classes and existing images
        self.samples = []
        for item in data['data']:
            # Use first class as primary label
            if item['image_classes'] and item['image_classes'][0] in class_to_idx:
                img_path = self.images_dir / item['image_path']
                self.samples.append({
                    'image_path': str(img_path),
                    'image_id': item['image_id'],
                    'label': class_to_idx[item['image_classes'][0]],
                    'caption': item['caption_str'],
                })
        
        if max_samples:
            self.samples = self.samples[:max_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Load image
        try:
            img = Image.open(item['image_path']).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            # Return a placeholder if image doesn't exist yet
            img = torch.zeros(3, 224, 224)
        
        label = item['label']
        group = 0  # Visual group
        
        return img, label, group


class TextCapsTextDataset(Dataset):
    """TextCaps dataset - Text/OCR modality (Group 1).
    
    Returns encoded caption text with class labels.
    """
    
    def __init__(
        self,
        annotations_path: str,
        class_to_idx: Dict[str, int],
        text_encoder: SimpleTextEncoder,
        max_samples: Optional[int] = None,
    ):
        self.text_encoder = text_encoder
        self.class_to_idx = class_to_idx
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        # Filter to samples with valid classes
        self.samples = []
        for item in data['data']:
            if item['image_classes'] and item['image_classes'][0] in class_to_idx:
                self.samples.append({
                    'image_id': item['image_id'],
                    'label': class_to_idx[item['image_classes'][0]],
                    'caption': item['caption_str'],
                })
        
        if max_samples:
            self.samples = self.samples[:max_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Encode text
        text_tensor = self.text_encoder.encode(item['caption'])
        
        label = item['label']
        group = 1  # Text group
        
        return text_tensor, label, group


class TextCapsMultiModalDataset(Dataset):
    """Combined multi-modal TextCaps dataset.
    
    Each image appears twice - once as visual features (group 0),
    once as text features (group 1).
    """
    
    def __init__(
        self,
        annotations_path: str,
        images_dir: str,
        class_to_idx: Dict[str, int],
        text_encoder: SimpleTextEncoder,
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.text_encoder = text_encoder
        self.class_to_idx = class_to_idx
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        # Build samples list - each image creates TWO samples
        # Only include samples where image file exists!
        self.samples = []
        count = 0
        skipped = 0
        for item in data['data']:
            if item['image_classes'] and item['image_classes'][0] in class_to_idx:
                label = class_to_idx[item['image_classes'][0]]
                img_path = self.images_dir / item['image_path']
                
                # Skip if image doesn't exist
                if not img_path.exists():
                    skipped += 1
                    continue
                
                # Visual sample (group 0)
                self.samples.append({
                    'modality': 'visual',
                    'image_path': str(img_path),
                    'image_id': item['image_id'],
                    'label': label,
                    'group': 0,
                })
                
                # Text sample (group 1)
                self.samples.append({
                    'modality': 'text',
                    'caption': item['caption_str'],
                    'image_id': item['image_id'],
                    'label': label,
                    'group': 1,
                })
                
                count += 1
                if max_samples and count >= max_samples:
                    break
        
        if skipped > 0:
            print(f"Skipped {skipped} samples (images not found)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        if item['modality'] == 'visual':
            try:
                img = Image.open(item['image_path']).convert('RGB')
                x = self.transform(img)
            except Exception:
                x = torch.zeros(3, 224, 224)
        else:
            x = self.text_encoder.encode(item['caption'])
        
        return x, item['label'], item['group']


# ---------------------------------------------------------------------------
# Collate Function for Mixed Modalities
# ---------------------------------------------------------------------------

def textcaps_collate_fn(batch):
    """Collate function that handles mixed visual, text, and combined samples.
    
    Visual samples: (C, H, W) tensors
    Text samples: (seq_len,) token tensors
    Combined samples: ((C, H, W), (seq_len,)) tuple of tensors
    
    We keep them separate and return a dict.
    """
    xs, ys, gs = zip(*batch)
    
    # Separate by group
    visual_indices = [i for i, g in enumerate(gs) if g == 0]
    text_indices = [i for i, g in enumerate(gs) if g == 1]
    combined_indices = [i for i, g in enumerate(gs) if g == 2]
    
    # Stack visual samples
    if visual_indices:
        visual_x = torch.stack([xs[i] for i in visual_indices])
        visual_y = torch.tensor([ys[i] for i in visual_indices], dtype=torch.long)
        visual_g = torch.zeros(len(visual_indices), dtype=torch.long)
    else:
        visual_x = torch.empty(0, 3, 224, 224)
        visual_y = torch.empty(0, dtype=torch.long)
        visual_g = torch.empty(0, dtype=torch.long)
    
    # Stack text samples
    if text_indices:
        text_x = torch.stack([xs[i] for i in text_indices])
        text_y = torch.tensor([ys[i] for i in text_indices], dtype=torch.long)
        text_g = torch.ones(len(text_indices), dtype=torch.long)
    else:
        text_x = torch.empty(0, 128, dtype=torch.long)  # max_len=128
        text_y = torch.empty(0, dtype=torch.long)
        text_g = torch.empty(0, dtype=torch.long)
    
    # Stack combined samples (visual+text pairs)
    if combined_indices:
        # xs[i] is a tuple (visual_tensor, text_tensor) for combined samples
        combined_visual_x = torch.stack([xs[i][0] for i in combined_indices])
        combined_text_x = torch.stack([xs[i][1] for i in combined_indices])
        combined_y = torch.tensor([ys[i] for i in combined_indices], dtype=torch.long)
        combined_g = torch.full((len(combined_indices),), 2, dtype=torch.long)
    else:
        combined_visual_x = torch.empty(0, 3, 224, 224)
        combined_text_x = torch.empty(0, 128, dtype=torch.long)
        combined_y = torch.empty(0, dtype=torch.long)
        combined_g = torch.empty(0, dtype=torch.long)
    
    return {
        'visual_x': visual_x,
        'visual_y': visual_y,
        'visual_g': visual_g,
        'text_x': text_x,
        'text_y': text_y,
        'text_g': text_g,
        'combined_visual_x': combined_visual_x,
        'combined_text_x': combined_text_x,
        'combined_y': combined_y,
        'combined_g': combined_g,
    }


# ---------------------------------------------------------------------------
# Loader Builders
# ---------------------------------------------------------------------------

def get_textcaps_class_mapping(
    annotations_path: str,
    top_k: Optional[int] = 10,
) -> Tuple[Dict[str, int], List[str]]:
    """Get class-to-index mapping for top K most frequent classes (local JSON).
    
    Args:
        annotations_path: Path to JSON annotations file
        top_k: Number of top classes to use. If None, uses ALL classes.
    
    Returns:
        class_to_idx: Dictionary mapping class name to index
        class_names: List of class names in order
    """
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    # Count primary classes (first in list)
    class_counts = Counter()
    for item in data['data']:
        if item['image_classes']:
            class_counts[item['image_classes'][0]] += 1
    
    # Get top K or all classes
    if top_k is None:
        # Use ALL classes, sorted by frequency
        all_classes = [cls for cls, _ in class_counts.most_common()]
    else:
        # Get top K
        all_classes = [cls for cls, _ in class_counts.most_common(top_k)]
    
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    
    return class_to_idx, all_classes


def get_textcaps_class_mapping_hf(
    hf_dataset,
    top_k: Optional[int] = 10,
    class_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, int], List[str]]:
    """Get class-to-index mapping for top K most frequent classes or explicit class list (Hugging Face).
    
    Args:
        hf_dataset: Hugging Face dataset
        top_k: Number of top classes to use. If None, uses ALL classes. Ignored if class_names is set.
        class_names: If provided, use exactly these classes (e.g. ["Bottle", "Car", "Food", "Book"]).
    
    Returns:
        class_to_idx: Dictionary mapping class name to index
        class_names: List of class names in order
    """
    if class_names is not None:
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        return class_to_idx, list(class_names)

    class_counts = Counter()
    for item in hf_dataset:
        image_classes = item.get('image_classes', [])
        if image_classes:
            class_counts[image_classes[0]] += 1

    # Get top K or all classes
    if top_k is None:
        # Use ALL classes, sorted by frequency
        all_classes = [cls for cls, _ in class_counts.most_common()]
    else:
        # Get top K
        all_classes = [cls for cls, _ in class_counts.most_common(top_k)]

    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}

    return class_to_idx, all_classes


def build_textcaps_loaders_hf(
    batch_size: int,
    num_workers: int = 0,
    num_classes: Optional[int] = 10,
    class_names: Optional[List[str]] = None,
    train_frac: Optional[float] = None,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 1337,
    include_combined: bool = False,
    use_stratified_sampling: bool = True,
    spurious_correlation: bool = False,
    spurious_majority_frac: float = 0.8,
    spurious_group_classes: Optional[Dict[int, List[int]]] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], "SimpleTextEncoder", Dict[str, any]]:
    """Build TextCaps data loaders from Hugging Face.
    
    This is the recommended method - images are hosted on HF, no Flickr issues.
    
    Args:
        batch_size: Batch size
        num_workers: DataLoader workers
        num_classes: Number of top classes to use. If None, uses ALL classes. Ignored if class_names is set.
        class_names: If set, use exactly these classes (e.g. ["Bottle", "Car", "Food", "Book"] for 4-class).
        train_frac: If set (e.g. 0.8), merge train+val and split 80%% train / (1-train_frac) test. Else use HF train/val as-is.
        max_train_samples: Limit training samples (per modality pair)
        max_test_samples: Limit test samples (per modality pair)
        seed: Random seed
        include_combined: If True, include third group (visual+text combined)
        use_stratified_sampling: If True, use stratified batch sampler to maintain π distribution
        spurious_correlation: If True, create class-group correlations in training data
        spurious_majority_frac: Fraction of samples from majority class(es) per group (default 0.8)
        spurious_group_classes: Dict mapping group_id -> list of majority class indices.
            Default for 4-class: {0: [0], 1: [1,2], 2: [3]} meaning:
            - Group 0 (visual): 80% Bottle
            - Group 1 (text): 80% Car+Food
            - Group 2 (combined): 80% Book
        
    Returns:
        train_loader, test_loader, class_to_idx, text_encoder, dataset_info
        
        dataset_info contains:
            - train_group_counts: Dict[int, int] mapping group_id to sample count
            - test_group_counts: Dict[int, int]
            - train_total: Total training samples
            - test_total: Total test samples
            - class_names: List of class names
            - num_groups: Number of groups (2 or 3)
    """
    if not HF_AVAILABLE:
        raise ImportError("huggingface datasets library required. Install with: pip install datasets")

    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Loading TextCaps from Hugging Face...")
    ds = load_dataset('lmms-lab/TextCaps')
    print(f"Loaded: {len(ds['train'])} train, {len(ds['val'])} val samples")

    # Get class mapping: explicit class names (e.g. 4-class) or top_k from train
    class_to_idx, class_names_out = get_textcaps_class_mapping_hf(
        ds['train'], top_k=num_classes, class_names=class_names
    )
    if class_names is not None:
        print(f"Using explicit classes ({len(class_names_out)}): {class_names_out}")
    elif num_classes is None:
        print(f"Using ALL {len(class_names_out)} classes")
    else:
        print(f"Using top {num_classes} classes: {class_names_out}")

    # Create text encoder
    text_encoder = SimpleTextEncoder(max_len=128, embed_dim=64)

    if train_frac is not None:
        # Merge train+val, filter to our classes, then 80/20 split
        full = concatenate_datasets([ds['train'], ds['val']])
        # Filter to samples whose primary class is in class_to_idx
        def in_classes(item):
            ic = item.get('image_classes') or []
            return bool(ic and ic[0] in class_to_idx)
        full_filtered = full.filter(in_classes)
        split = full_filtered.train_test_split(test_size=1.0 - train_frac, seed=seed)
        hf_train, hf_test = split['train'], split['test']
        print(f"80/20 split: {len(hf_train)} train, {len(hf_test)} test (after filtering to {len(class_to_idx)} classes)")
    else:
        hf_train, hf_test = ds['train'], ds['val']

    # Create datasets
    # Training: optionally with spurious correlation
    train_ds = TextCapsHFDataset(
        hf_dataset=hf_train,
        class_to_idx=class_to_idx,
        text_encoder=text_encoder,
        max_samples=max_train_samples,
        include_combined=include_combined,
        spurious_correlation=spurious_correlation,
        spurious_majority_frac=spurious_majority_frac,
        spurious_group_classes=spurious_group_classes,
        seed=seed,
    )

    # Test: always balanced (no spurious correlation)
    test_ds = TextCapsHFDataset(
        hf_dataset=hf_test,
        class_to_idx=class_to_idx,
        text_encoder=text_encoder,
        max_samples=max_test_samples,
        include_combined=include_combined,
        spurious_correlation=False,  # Test set is always balanced
        seed=seed + 1,
    )

    samples_per_image = 3 if include_combined else 2
    if spurious_correlation:
        print(f"Train samples: {len(train_ds)} (spurious correlation mode)")
        # Print class distribution per group
        train_class_dist = train_ds.get_class_distribution_per_group()
        group_names = ['visual', 'text', 'combined'] if include_combined else ['visual', 'text']
        print("Train class distribution per group (spurious correlation):")
        for g, class_counts in sorted(train_class_dist.items()):
            total = sum(class_counts.values())
            class_pcts = {c: f"{cnt/total*100:.1f}%" for c, cnt in sorted(class_counts.items())}
            print(f"  Group {g} ({group_names[g]}): {class_pcts} (n={total})")
    else:
        print(f"Train samples: {len(train_ds)} ({samples_per_image}x per image for multi-modal)")
    print(f"Test samples: {len(test_ds)} ({samples_per_image}x per image, balanced)")
    
    # Print group distribution
    print(f"Train group counts: {train_ds.group_counts}")
    print(f"Test group counts: {test_ds.group_counts}")
    
    # Create loaders (pin_memory=False when num_workers=0 to avoid Mac/MPS issues)
    pin_memory = num_workers > 0
    
    # Use stratified sampling for training if enabled
    if use_stratified_sampling:
        train_sampler = StratifiedGroupSampler(
            group_labels=train_ds.get_group_labels(),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=textcaps_collate_fn,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=textcaps_collate_fn,
            pin_memory=pin_memory,
        )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=textcaps_collate_fn,
        pin_memory=pin_memory,
    )
    
    # Compute per-class sample counts
    train_class_counts = Counter()
    test_class_counts = Counter()
    if spurious_correlation:
        # In spurious mode, each sample is unique (1 sample per image)
        for sample in train_ds.samples:
            train_class_counts[sample['label']] += 1
    else:
        # In standard mode, count once per image (use group 0 as proxy)
        for sample in train_ds.samples:
            if sample['group'] == 0:
                train_class_counts[sample['label']] += 1
    for sample in test_ds.samples:
        if sample['group'] == 0:
            test_class_counts[sample['label']] += 1
    
    # Build dataset info dict
    train_images = len(train_ds) if spurious_correlation else len(train_ds) // samples_per_image
    test_images = len(test_ds) // samples_per_image
    
    dataset_info = {
        'train_group_counts': train_ds.group_counts,
        'test_group_counts': test_ds.group_counts,
        'train_total': len(train_ds),
        'test_total': len(test_ds),
        'train_images': train_images,
        'test_images': test_images,
        'class_names': class_names_out,
        'num_groups': 3 if include_combined else 2,
        'group_names': ['visual', 'text', 'combined'] if include_combined else ['visual', 'text'],
        'train_class_counts': dict(train_class_counts),
        'test_class_counts': dict(test_class_counts),
        'spurious_correlation': spurious_correlation,
        'spurious_majority_frac': spurious_majority_frac if spurious_correlation else None,
        'train_class_dist_per_group': train_ds.get_class_distribution_per_group() if spurious_correlation else None,
    }
    
    return train_loader, test_loader, class_to_idx, text_encoder, dataset_info


def build_textcaps_loaders(
    root: str,
    batch_size: int,
    num_workers: int = 0,
    num_classes: Optional[int] = 10,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 1337,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], SimpleTextEncoder]:
    """Build TextCaps data loaders for multi-modal GroupDRO.
    
    Args:
        root: Path to datasets directory
        batch_size: Batch size
        num_workers: DataLoader workers
        num_classes: Number of top classes to use
        max_train_samples: Limit training samples (per modality)
        max_test_samples: Limit test samples (per modality)
        seed: Random seed
        
    Returns:
        train_loader, test_loader, class_to_idx, text_encoder
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    textcaps_dir = Path(root) / 'textcaps'
    annotations_dir = textcaps_dir / 'annotations'
    images_dir = textcaps_dir / 'images'
    
    train_ann = annotations_dir / 'TextCaps_0.1_train.json'
    val_ann = annotations_dir / 'TextCaps_0.1_val.json'
    
    # Get class mapping from training data
    class_to_idx, class_names = get_textcaps_class_mapping(
        str(train_ann), top_k=num_classes
    )
    if num_classes is None:
        print(f"Using ALL {len(class_names)} classes")
    else:
        print(f"Using top {num_classes} classes: {class_names}")
    
    # Create text encoder
    text_encoder = SimpleTextEncoder(max_len=128, embed_dim=64)
    
    # Create datasets
    # Note: image_path in annotations already includes 'train/' or 'val/' prefix
    train_ds = TextCapsMultiModalDataset(
        annotations_path=str(train_ann),
        images_dir=str(images_dir),  # Don't add 'train' - it's in image_path
        class_to_idx=class_to_idx,
        text_encoder=text_encoder,
        max_samples=max_train_samples,
    )
    
    test_ds = TextCapsMultiModalDataset(
        annotations_path=str(val_ann),
        images_dir=str(images_dir),  # Don't add 'val' - it's in image_path
        class_to_idx=class_to_idx,
        text_encoder=text_encoder,
        max_samples=max_test_samples,
    )
    
    print(f"Train samples: {len(train_ds)} (2x per image for multi-modal)")
    print(f"Test samples: {len(test_ds)}")
    
    # Create loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=textcaps_collate_fn,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=textcaps_collate_fn,
        pin_memory=True,
    )
    
    return train_loader, test_loader, class_to_idx, text_encoder


# ---------------------------------------------------------------------------
# Image Download Helper
# ---------------------------------------------------------------------------

def create_image_download_script(
    annotations_path: str,
    output_script: str,
    class_to_idx: Dict[str, int],
    max_images: Optional[int] = None,
):
    """Create a shell script to download images from Flickr URLs.
    
    Note: Images are from OpenImages via Flickr. We use the 300k URLs
    which are smaller and faster to download.
    """
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    script_lines = [
        "#!/bin/bash",
        "# TextCaps image download script",
        "# Run with: bash download_textcaps_images.sh",
        "",
        "mkdir -p datasets/textcaps/images/train",
        "mkdir -p datasets/textcaps/images/val",
        "",
    ]
    
    count = 0
    for item in data['data']:
        if item['image_classes'] and item['image_classes'][0] in class_to_idx:
            url = item.get('flickr_300k_url', item.get('flickr_original_url'))
            if url:
                out_path = f"datasets/textcaps/images/{item['image_path']}"
                script_lines.append(f'curl -s -o "{out_path}" "{url}" &')
                count += 1
                
                # Batch downloads (10 at a time)
                if count % 10 == 0:
                    script_lines.append("wait")
                
                if max_images and count >= max_images:
                    break
    
    script_lines.append("wait")
    script_lines.append(f'echo "Downloaded {count} images"')
    
    with open(output_script, 'w') as f:
        f.write('\n'.join(script_lines))
    
    print(f"Created download script: {output_script}")
    print(f"Will download {count} images")
    print(f"Run with: bash {output_script}")


if __name__ == "__main__":
    # Test the loader
    import sys
    
    root = "datasets"
    
    # Get class mapping first
    class_to_idx, class_names = get_textcaps_class_mapping(
        f"{root}/textcaps/annotations/TextCaps_0.1_train.json",
        top_k=10
    )
    print(f"Classes: {class_names}")
    
    # Create download script
    create_image_download_script(
        f"{root}/textcaps/annotations/TextCaps_0.1_train.json",
        "download_textcaps_images.sh",
        class_to_idx,
        max_images=1000,  # Start small
    )
