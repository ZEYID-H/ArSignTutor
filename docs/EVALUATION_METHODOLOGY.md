# Evaluation Methodology

---

## Metrics

All metrics are computed using `sklearn.metrics` on the held-out test split.

| Metric | Formula | Why used |
|--------|---------|---------|
| **Accuracy** | correct / total | Primary metric; interpretable for single-label classification |
| **Macro Precision** | mean per-class precision | Penalizes false positives equally across all classes |
| **Macro Recall** | mean per-class recall | Penalizes missed signs equally across all classes |
| **Macro F1** | harmonic mean of precision & recall | Balanced metric; preferred over accuracy when classes are unequal size |

Macro averaging is used because all 31 letters carry equal importance — a model that fails on rare letters isn't usable.

---

## Evaluation Scripts

### In-Distribution Evaluation

Tests on the held-out test split (same distribution as training, but cluster-isolated):

```bash
python CODE/src/evaluation/evaluate.py
```

**Outputs:**
- Accuracy, precision, recall, F1 printed to console
- `classification_report.csv` — per-class breakdown
- `confusion_matrix.png` — 31×31 heatmap

### Cross-Dataset Evaluation

Tests a model trained on one dataset against a completely different dataset:

```bash
python CODE/evaluate_cross_dataset.py \
    --model CODE/arsigntutor_final_results/best_rgb_domain_adapted_mobilenetv2.pth \
    --dataset dataset/arasl/ \
    --output results/cross_dataset/
```

**Purpose:** If in-distribution accuracy is 94% but cross-dataset accuracy drops to 50%, the model has learned dataset-specific artifacts rather than sign shapes. This is the key diagnostic for data leakage.

---

## Data Leakage Case Study

### Timeline

```
Experiment 1  →  99.26% accuracy   [naive random split on ArASL 54K]
     ↓
Suspicion: Is this too good?
     ↓
Experiment 2  →  48.78% accuracy   [cross-dataset test on RGB Alphabets]
     ↓
Root cause: Near-duplicate video frames contaminate train/test
     ↓
Fix: dHash clustering → cluster-aware split
     ↓
Experiment 3  →  94.33% accuracy   [clean retrain on RGB with cluster split]
```

### Experiment 1: The Suspicious Result

Random 70/15/15 split on ArASL 54K with MobileNetV2:

| Split | Accuracy |
|-------|---------|
| Train | 99.8% |
| Validation | 99.1% |
| **Test** | **99.26%** |

The near-zero train/test gap on a 15,000-image test set was the first warning sign.

### Experiment 2: Cross-Dataset Reality Check

The same model tested on the RGB Arabic Alphabets dataset (different signers, backgrounds, lighting):

| | Accuracy |
|--|---------|
| Cross-dataset average | **48.78%** |

A 50.5-point gap is a near-certain indicator of data leakage.

### Root Cause Analysis

ArASL was collected as continuous video. A single recording of one signer doing one letter produces a burst of frames. Consecutive frames in a burst are near-pixel-identical — they differ only by noise and slight finger position variations.

When split randomly, the same signer performing the same sign appears in both train and test:

```
train/seen/signer_01_frame_001.jpg  ←─┐ Near-identical
test/seen/signer_01_frame_003.jpg   ←─┘ (2-frame apart)
```

The model memorized signer appearance instead of sign shapes — hence the collapse when signers change.

### Verification: dHash Confirms the Contamination

A dHash analysis of the ArASL dataset revealed:

- Thousands of image clusters where all members are near-identical (Hamming distance ≤ 8)
- Clusters correspond to recording bursts (consecutive frame ranges)
- With random splitting, ~13% of test images have a near-identical twin in the training set

### Experiment 3: Clean Evaluation After Leakage-Aware Split

**Setup:**
- Dataset: RGB Arabic Alphabets Sign Language (fresh start, different source)
- Split: dHash cluster-aware (see DATASET_PREPARATION.md)
- Model: Same MobileNetV2 architecture, retrained from ImageNet weights

**Results:**

| Metric | Value |
|--------|-------|
| Test Accuracy | **94.33%** |
| Macro Precision | 94.59% |
| Macro Recall | 94.33% |
| Macro F1 | 94.36% |
| Best Validation | 93.32% |

The 4.93% gap between clean accuracy (94.33%) and leaked accuracy (99.26%) represents the "leakage premium" — the artificial boost caused by contaminated test data.

The ~6% error rate concentrates on visually similar pairs (dal/thal, seen/sheen, ta/dha) — expected failures, not model defects.

---

## Per-Class Analysis

### Top Performing Classes

| Letter | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| ha | 1.00 | 1.00 | 1.00 |
| jeem | 1.00 | 1.00 | 1.00 |
| ghain | 1.00 | 1.00 | 1.00 |
| ain | 0.97 | 1.00 | 0.99 |
| waw | 0.98 | 0.97 | 0.97 |

### Most Common Confusion Pairs

| Predicted | True | Reason |
|-----------|------|--------|
| dal | dhal | Near-identical hand shapes; differ only in finger angle |
| seen | sheen | Similar palm orientation; differ in finger spread |
| ta | dha | Both involve bent wrist; subtle thumb position difference |
| fa | ghain | Similar closed-fist base; differ in index finger extension |

The app's 70% confidence threshold mitigates most of these at inference time.

---

## Inference-Time Evaluation

The Streamlit application uses **temporal smoothing** during live webcam inference, which functions as an implicit ensemble:

```
25 consecutive frames → majority vote → final prediction
(minimum 15 votes required for a confident prediction)
```

Single-frame predictions are noisy; correct predictions tend to be consistent while errors cancel out in the vote. In practice, real-world accuracy exceeds the 94.33% single-frame test number.

---

## Reproducing the Evaluation

To reproduce the published test results exactly:

```bash
# 1. Ensure the RGB dataset is at dataset/rgb_arabic/
# 2. Use the committed split files (do not re-split)
python CODE/src/evaluation/evaluate.py \
    --model CODE/arsigntutor_final_results/best_rgb_domain_adapted_mobilenetv2.pth \
    --test-split CODE/arsigntutor_final_results/.../splits/rgb_test_files.json

# Expected output:
# Accuracy:  94.33% (1093/1165)
# F1 Macro:  94.36%
```

To reproduce the cross-dataset evaluation:

```bash
python CODE/evaluate_cross_dataset.py \
    --model CODE/arsigntutor_final_results/best_rgb_domain_adapted_mobilenetv2.pth \
    --dataset dataset/arasl/ \
    --mode cross_dataset

# Tests the original leaky model (99.26%) on the RGB dataset
# Expected output: ~48.78% accuracy
```
