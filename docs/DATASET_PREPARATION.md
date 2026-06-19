# Dataset Preparation Guide

How to obtain, validate, and split the datasets used in ArSignTutor.

---

## Datasets Used

| Dataset | Role | Source |
|---------|------|--------|
| RGB Arabic Alphabets Sign Language | Primary training data | Kaggle |
| ArASL Database 54K | Cross-dataset generalization test | Kaggle |

---

## 1. Download Datasets

### RGB Arabic Alphabets Sign Language

```bash
kaggle datasets download -d monasaleh/arabic-alphabets-sign-language-dataset
unzip arabic-alphabets-sign-language-dataset.zip -d dataset/rgb_arabic/
```

**Expected structure:**
```
dataset/rgb_arabic/
├── ain/
├── al/
├── aleff/
├── bb/
├── dal/
... (31 class directories)
```

### ArASL Database 54K (for cross-dataset evaluation only)

```bash
kaggle datasets download -d ammarsayedtaha/arabic-sign-language-dataset-2022
unzip arabic-sign-language-dataset-2022.zip -d dataset/arasl/
```

---

## 2. Dataset Validation

Before splitting, verify that all classes are present and image files are readable:

```bash
python CODE/src/preprocessing/check_dataset.py --dataset dataset/rgb_arabic/
```

This script checks:
- All 31 expected class directories exist
- No corrupted or zero-byte image files
- Per-class sample counts (flags severe imbalance)

**Expected output:**
```
[OK] 31/31 classes found
[OK] 1,247 total images
[INFO] Min class size: 28 (khaa)
[INFO] Max class size: 52 (aleff)
[WARN] Class imbalance ratio: 1.86x (acceptable)
```

---

## 3. The Data Leakage Problem

### Why Random Splitting Fails for Video-Derived Datasets

The ArASL 54K dataset (and many similar datasets) is recorded as continuous video. A single recording session of one signer performing one letter produces hundreds of nearly identical frames:

```
signer_01_seen_frame_001.jpg  ─┐
signer_01_seen_frame_002.jpg   │ All near-duplicates from
signer_01_seen_frame_003.jpg   │ the same 2-second video clip
...                            │
signer_01_seen_frame_047.jpg  ─┘
```

When these are split randomly:
- ~70% of frames go to train
- ~15% go to val
- ~15% go to test

The test set contains frames that are **pixel-level near-identical** to training frames. The model effectively memorizes signers rather than sign shapes, producing inflated accuracy.

**Evidence:** A model trained with random splitting on ArASL achieved **99.26% test accuracy** but only **48.78% accuracy** on the completely independent RGB dataset.

---

## 4. dHash Duplicate Detection

**Difference hashing (dHash)** detects near-duplicate images by comparing pixel intensity gradients:

### Algorithm

```python
import cv2
import numpy as np

def dhash(image_path: str, hash_size: int = 8) -> int:
    """Compute 64-bit dHash for an image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (hash_size + 1, hash_size))
    diff = img[:, 1:] > img[:, :-1]          # horizontal gradient directions
    return sum(2**i for i, v in enumerate(diff.flatten()) if v)

def hamming_distance(hash1: int, hash2: int) -> int:
    return bin(hash1 ^ hash2).count('1')

# Images are near-duplicates if Hamming distance <= threshold
THRESHOLD = 8   # out of 64 bits; ~12.5% difference tolerance
```

### Clustering Process

```python
def cluster_by_dhash(image_paths, threshold=8, window=100):
    """
    Group images into clusters where all members are near-duplicates.
    Uses a sliding window for efficiency on large datasets.
    """
    hashes = {p: dhash(p) for p in image_paths}
    clusters = {}
    cluster_id = 0

    for i, path_i in enumerate(image_paths):
        if path_i in clusters:
            continue
        clusters[path_i] = cluster_id
        # Compare against nearby images (same recording session = adjacent filenames)
        for path_j in image_paths[i+1 : i+window]:
            if hamming_distance(hashes[path_i], hashes[path_j]) <= threshold:
                clusters[path_j] = cluster_id
        cluster_id += 1

    return clusters
```

### Cluster-Aware Splitting

Once images are grouped into clusters, **entire clusters** are assigned to a single split:

```python
def cluster_aware_split(clusters, ratios=(0.70, 0.15, 0.15), seed=42):
    """
    Assign each cluster entirely to train, val, or test.
    No cluster can span multiple splits.
    """
    unique_clusters = list(set(clusters.values()))
    random.seed(seed)
    random.shuffle(unique_clusters)

    n = len(unique_clusters)
    train_end = int(n * ratios[0])
    val_end   = int(n * (ratios[0] + ratios[1]))

    train_clusters = set(unique_clusters[:train_end])
    val_clusters   = set(unique_clusters[train_end:val_end])
    test_clusters  = set(unique_clusters[val_end:])

    return {
        path: ('train' if cid in train_clusters else
               'val'   if cid in val_clusters   else 'test')
        for path, cid in clusters.items()
    }
```

---

## 5. Running the Split

```bash
python CODE/src/preprocessing/split_dataset.py \
    --dataset dataset/rgb_arabic/ \
    --output CODE/arsigntutor_final_results/final_project_package/results_clean_training/splits/ \
    --train 0.70 --val 0.15 --test 0.15 \
    --hash-threshold 8 --window 100 \
    --seed 42
```

### Verifying the Split

```bash
python CODE/src/preprocessing/check_split.py \
    --splits CODE/arsigntutor_final_results/.../splits/
```

Checks:
- No image appears in more than one split
- No cluster is split across train/val/test
- Split ratios are approximately correct
- All 31 classes are represented in each split

---

## 6. Pre-computed Split Files

The split files used for the published results are committed to the repository:

```
CODE/arsigntutor_final_results/final_project_package/results_clean_training/splits/
├── rgb_train_files.json         # List of training image paths
├── rgb_val_files.json           # List of validation image paths
├── rgb_test_files.json          # List of test image paths
└── rgb_cluster_split_all.csv    # Full table: path, cluster_id, split
```

To reproduce the exact evaluation without re-splitting:

```python
import json
with open('.../rgb_test_files.json') as f:
    test_files = json.load(f)
# Use test_files directly as your test set
```

---

## 7. Dataset Statistics

After leakage-aware splitting on the RGB Arabic Alphabets dataset:

| Split | Images | % of Total |
|-------|--------|-----------|
| Train | ~875 | 70% |
| Validation | ~186 | 15% |
| Test | ~186 | 15% |
| **Total** | **~1,247** | 100% |

Classes: 31 Arabic letters (aleff, bb, ta, tha, jeem, haa, khaa, dal, thal, ra, zay, seen, sheen, saad, dhad, toot, dha, ain, ghain, fa, gaaf, kaaf, la, laam, meem, nun, ha, waw, ya, aleff variants)

> Note: `yaa` is excluded from training due to insufficient or ambiguous samples in this dataset.
