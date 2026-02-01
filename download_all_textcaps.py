#!/usr/bin/env python3
"""
Download all TextCaps images from annotations.
Uses parallel downloads for speed.
"""
import json
import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

def download_image(args):
    """Download a single image."""
    url, output_path = args
    if os.path.exists(output_path):
        return True, output_path, "exists"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True, output_path, "downloaded"
        else:
            return False, output_path, f"status {response.status_code}"
    except Exception as e:
        return False, output_path, str(e)

def extract_urls_from_annotations(ann_path, images_dir, split='train'):
    """Extract image URLs from TextCaps annotations."""
    urls = []
    
    with open(ann_path, 'r') as f:
        data = json.load(f)
    
    # TextCaps format has 'data' list with 'image_path' and 'flickr_original_url'
    for item in data.get('data', []):
        if 'flickr_original_url' in item and 'image_path' in item:
            url = item['flickr_original_url']
            # image_path is like "train/xxxxx.jpg"
            img_path = item['image_path']
            output_path = os.path.join(images_dir, img_path)
            urls.append((url, output_path))
    
    return urls

def main():
    base_dir = Path(__file__).parent
    ann_dir = base_dir / 'datasets' / 'textcaps' / 'annotations'
    images_dir = base_dir / 'datasets' / 'textcaps' / 'images'
    
    # Collect all URLs from train and val
    all_urls = []
    
    train_ann = ann_dir / 'TextCaps_0.1_train.json'
    val_ann = ann_dir / 'TextCaps_0.1_val.json'
    
    if train_ann.exists():
        print(f"Extracting URLs from {train_ann}...")
        train_urls = extract_urls_from_annotations(train_ann, images_dir, 'train')
        print(f"  Found {len(train_urls)} training images")
        all_urls.extend(train_urls)
    
    if val_ann.exists():
        print(f"Extracting URLs from {val_ann}...")
        val_urls = extract_urls_from_annotations(val_ann, images_dir, 'val')
        print(f"  Found {len(val_urls)} validation images")
        all_urls.extend(val_urls)
    
    print(f"\nTotal images to download: {len(all_urls)}")
    
    # Check how many already exist
    existing = sum(1 for url, path in all_urls if os.path.exists(path))
    print(f"Already downloaded: {existing}")
    print(f"Need to download: {len(all_urls) - existing}")
    
    if len(all_urls) - existing == 0:
        print("All images already downloaded!")
        return
    
    # Download in parallel
    print(f"\nStarting parallel download with 50 workers...")
    start_time = time.time()
    
    success = 0
    failed = 0
    skipped = 0
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(download_image, args): args for args in all_urls}
        
        with tqdm(total=len(all_urls), desc="Downloading") as pbar:
            for future in as_completed(futures):
                ok, path, msg = future.result()
                if ok:
                    if msg == "exists":
                        skipped += 1
                    else:
                        success += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix({"ok": success, "skip": skipped, "fail": failed})
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Download complete in {elapsed/60:.1f} minutes")
    print(f"  Successfully downloaded: {success}")
    print(f"  Already existed: {skipped}")
    print(f"  Failed: {failed}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
