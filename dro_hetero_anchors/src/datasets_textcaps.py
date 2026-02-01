"""
TextCaps Dataset Loader for GroupDRO with Heterogeneous Feature Spaces

Multi-modal setup:
- Group 0: Visual features (CNN encoder on images)
- Group 1: Text features (text encoder on OCR/caption text)

Each image appears twice in the dataset - once per modality.
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image

# Try to import huggingface datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


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
    
    Each sample appears twice - once as visual (group 0) and once as text (group 1).
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
    ):
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.text_encoder = text_encoder
        self.class_to_idx = class_to_idx
        
        # Store reference to HF dataset for lazy loading
        self.hf_dataset = hf_dataset
        
        # Build samples list - each image creates TWO samples
        # IMPORTANT: Don't store PIL images! Store indices for lazy loading.
        self.samples = []
        count = 0
        skipped = 0
        
        for hf_idx, item in enumerate(hf_dataset):
            # Get primary class
            image_classes = item.get('image_classes', [])
            if image_classes and image_classes[0] in class_to_idx:
                label = class_to_idx[image_classes[0]]
                
                # Visual sample (group 0) - store HF index, NOT the image!
                # caption_str is a list in HF dataset - use first caption
                caption_field = item.get('caption_str', item.get('reference_strs', ['']))
                if isinstance(caption_field, list):
                    caption = caption_field[0] if caption_field else ''
                else:
                    caption = str(caption_field)
                
                self.samples.append({
                    'modality': 'visual',
                    'hf_idx': hf_idx,  # Store index for lazy loading
                    'image_id': item.get('image_id', str(count)),
                    'label': label,
                    'group': 0,
                })
                
                # Text sample (group 1)
                self.samples.append({
                    'modality': 'text',
                    'caption': caption,
                    'image_id': item.get('image_id', str(count)),
                    'label': label,
                    'group': 1,
                })
                
                count += 1
                if max_samples and count >= max_samples:
                    break
            else:
                skipped += 1
        
        if skipped > 0:
            print(f"Skipped {skipped} samples (no valid class or class not in mapping)")
        print(f"Created {len(self.samples)} samples from {count} images (lazy loading enabled)")
    
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
        else:
            x = self.text_encoder.encode(item['caption'])
        
        return x, item['label'], item['group']


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
    """Collate function that handles mixed visual and text samples.
    
    Visual samples: (C, H, W) tensors
    Text samples: (seq_len,) token tensors
    
    We keep them separate and return a dict.
    """
    xs, ys, gs = zip(*batch)
    
    # Separate by group
    visual_indices = [i for i, g in enumerate(gs) if g == 0]
    text_indices = [i for i, g in enumerate(gs) if g == 1]
    
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
    
    return {
        'visual_x': visual_x,
        'visual_y': visual_y,
        'visual_g': visual_g,
        'text_x': text_x,
        'text_y': text_y,
        'text_g': text_g,
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
) -> Tuple[Dict[str, int], List[str]]:
    """Get class-to-index mapping for top K most frequent classes (Hugging Face).
    
    Args:
        hf_dataset: Hugging Face dataset
        top_k: Number of top classes to use. If None, uses ALL classes.
    
    Returns:
        class_to_idx: Dictionary mapping class name to index
        class_names: List of class names in order
    """
    
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
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 1337,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], "SimpleTextEncoder"]:
    """Build TextCaps data loaders from Hugging Face.
    
    This is the recommended method - images are hosted on HF, no Flickr issues.
    
    Args:
        batch_size: Batch size
        num_workers: DataLoader workers
        num_classes: Number of top classes to use. If None, uses ALL classes.
        max_train_samples: Limit training samples (per modality pair)
        max_test_samples: Limit test samples (per modality pair)
        seed: Random seed
        
    Returns:
        train_loader, test_loader, class_to_idx, text_encoder
    """
    if not HF_AVAILABLE:
        raise ImportError("huggingface datasets library required. Install with: pip install datasets")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("Loading TextCaps from Hugging Face...")
    ds = load_dataset('lmms-lab/TextCaps')
    print(f"Loaded: {len(ds['train'])} train, {len(ds['val'])} val samples")
    
    # Get class mapping from training data
    class_to_idx, class_names = get_textcaps_class_mapping_hf(ds['train'], top_k=num_classes)
    if num_classes is None:
        print(f"Using ALL {len(class_names)} classes")
    else:
        print(f"Using top {num_classes} classes: {class_names}")
    
    # Create text encoder
    text_encoder = SimpleTextEncoder(max_len=128, embed_dim=64)
    
    # Create datasets
    train_ds = TextCapsHFDataset(
        hf_dataset=ds['train'],
        class_to_idx=class_to_idx,
        text_encoder=text_encoder,
        max_samples=max_train_samples,
    )
    
    test_ds = TextCapsHFDataset(
        hf_dataset=ds['val'],
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
