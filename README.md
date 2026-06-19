# ArSignTutor

**Interactive Real-Time Arabic Sign Language Learning System**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)](https://streamlit.io)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9%2B-brightgreen)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

ArSignTutor is a real-time Arabic Sign Language (ArSL) learning application that teaches users to spell Arabic words using hand signs detected via webcam. The system gamifies sign language acquisition through an interactive letter-by-letter spelling challenge, providing instant visual feedback powered by a fine-tuned MobileNetV2 classifier.

This project combines **rigorous ML evaluation practices** — including leakage-aware dataset splitting and cross-dataset generalization testing — with a production-ready Streamlit deployment suitable for real-world educational use.

---

## Key Achievement

> **Discovered that an apparent 99.26% test accuracy was caused by near-duplicate image leakage between train and test sets.**
>
> Cross-dataset evaluation revealed only **48.78%** real-world performance — a 50.5 percentage-point gap that exposed severe model memorization rather than generalization.
>
> Designed a **dHash-based clustering pipeline** to eliminate leakage, retrained the model from scratch on a clean split, and achieved an **honest 94.33% test accuracy** that holds up under independent evaluation.

This end-to-end data leakage discovery and resolution is the primary technical contribution of this project. It demonstrates the kind of evaluation integrity that separates production ML systems from academic benchmarks.

| Experiment | Split Strategy | Test Accuracy |
|------------|---------------|---------------|
| Baseline (ArASL 54K) | Random 70/15/15 | **99.26%** — inflated by leakage |
| Cross-dataset probe | N/A (different source) | **48.78%** — true generalization |
| Clean retrain (RGB dataset) | dHash cluster-aware | **94.33%** — honest result |

---

## Why This Project Matters

**Model generalization is the real test.** A classifier that memorizes training artifacts performs well on paper and fails in production. This project treats generalization as a first-class requirement, not an afterthought.

**Evaluation methodology is engineering.** The choice of how to split a dataset is not administrative — it determines whether your accuracy numbers mean anything. Near-duplicate video frames from the same recording session are a systematic contamination source in video-derived datasets, and most published ArSL benchmarks do not account for this.

**Data leakage detection is a transferable skill.** The dHash clustering approach developed here applies to any dataset collected from video: action recognition, medical imaging time-series, face recognition, sign language — wherever frames from the same session may land in different splits.

**Trustworthy AI requires honest metrics.** Reporting 94.33% instead of 99.26% is a deliberate choice that makes the system more trustworthy. The 5.67% accuracy that leakage "gave for free" would have misled every downstream decision about model readiness.

**Reproducibility is built in.** Split files are committed to the repository so the exact test set can be reproduced without re-running the clustering pipeline.

---

## Problem Statement

Arabic Sign Language is the primary communication method for approximately 50,000 deaf and hard-of-hearing individuals in Arab countries. Despite this, accessible and interactive learning tools remain scarce, creating a significant barrier to social inclusion.

Existing Arabic Sign Language recognition systems frequently suffer from **inflated accuracy metrics** caused by data leakage — where near-duplicate frames from the same recording session appear in both training and test splits. This leads to models that appear highly accurate in laboratory settings but generalize poorly to real users.

ArSignTutor addresses both the educational gap and the evaluation integrity problem.

---

## Motivation

1. **Build a practical learning tool** that makes Arabic Sign Language accessible to hearing learners
2. **Demonstrate rigorous ML evaluation** by identifying and correcting data leakage that inflated accuracy from 99.26% to an honest 94.33%
3. **Establish a reproducible baseline** for ArSL recognition using leakage-aware dataset splitting

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Web App                     │
│  ┌───────────────┐    ┌──────────────────────────────┐  │
│  │  Game Engine  │    │     Video Processing         │  │
│  │  - Word queue │    │     Pipeline                 │  │
│  │  - Scoring    │    │  ┌────────────────────────┐  │  │
│  │  - Streaks    │    │  │  MediaPipe             │  │  │
│  └───────────────┘    │  │  HandLandmarker        │  │  │
│                       │  │  (hand detection)      │  │  │
│                       │  └──────────┬─────────────┘  │  │
│                       │             │ Hand ROI        │  │
│                       │  ┌──────────▼─────────────┐  │  │
│                       │  │  MobileNetV2 Classifier │  │  │
│                       │  │  31 Arabic letters      │  │  │
│                       │  │  224×224 RGB input      │  │  │
│                       │  └──────────┬─────────────┘  │  │
│                       │             │ Prediction      │  │
│                       │  ┌──────────▼─────────────┐  │  │
│                       │  │  Temporal Smoothing     │  │  │
│                       │  │  25-frame buffer        │  │  │
│                       │  │  15-vote consensus      │  │  │
│                       │  └────────────────────────┘  │  │
│                       └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Stack:**

| Component | Technology |
|-----------|-----------|
| Model backbone | MobileNetV2 (ImageNet pre-trained) |
| Hand detection | MediaPipe HandLandmarker (0.10+ Tasks API) |
| Framework | PyTorch 2.0 |
| Web app | Streamlit + streamlit-webrtc (WebRTC) |
| Augmentation | Albumentations |
| Duplicate detection | dHash perceptual hashing (imagehash) |
| Evaluation | scikit-learn (classification report, confusion matrix) |

---

## Dataset Description

### Primary Dataset: RGB Arabic Alphabets Sign Language

- **Source:** RGB Arabic Alphabets Sign Language dataset (Kaggle)
- **Classes:** 31 Arabic letters (aleff, bb, ta, tha, jeem, haa, khaa, dal, thal, ra, zay, seen, sheen, saad, dhad, toot, dha, ain, ghain, fa, gaaf, kaaf, la, laam, meem, nun, ha, waw, ya, aleff variations)
- **Images:** ~1,200 samples across 31 classes
- **Format:** RGB JPEG images

### Reference Dataset: ArASL Database 54K

- **Source:** Arabic Alphabets Sign Language Dataset 2018 (54,000 images)
- **Role:** Used for cross-dataset generalization testing and data leakage investigation
- **Issue discovered:** Near-duplicate frames from the same recording session contaminate both train and test splits when using random splitting

---

## Data Leakage Discovery — A Case Study in Honest ML Evaluation

### Initial Results (Naive Random Split on ArASL)

A naive 70/15/15 random split on the ArASL 54K dataset produced the following metrics:

| Metric | Value |
|--------|-------|
| Test Accuracy | **99.26%** |
| Validation Accuracy | 99.1% |

These numbers appeared extraordinary — almost suspiciously so. The near-zero gap between train accuracy (99.8%) and test accuracy (99.26%) was the first warning sign.

### Cross-Dataset Validation Reveals the Problem

To test real-world generalization, the 99.26%-accurate model was evaluated on the **RGB Arabic Alphabets** dataset — a completely different source with different signers, lighting, and backgrounds:

| Metric | Value |
|--------|-------|
| Cross-Dataset Accuracy | **48.78%** |

A 50.5 percentage-point gap between in-distribution and out-of-distribution accuracy is a near-certain indicator of **data leakage**.

### Root Cause Analysis

The ArASL 54K dataset was recorded in controlled video sessions. Each session produces hundreds of nearly identical frames of the same signer performing the same sign. When samples are split **randomly**, frames from the same session appear in both the training set and the test set.

```
Session frame A (train):  signer_01_seen_frame_047.jpg  ┐
Session frame B (test):   signer_01_seen_frame_049.jpg  ┘  → 2-frame gap, near-identical pixels
```

The model learns to recognize **specific signers and recording artifacts** rather than generalizing sign shapes.

### dHash Duplicate Detection Methodology

**Difference hashing (dHash)** was applied to detect near-duplicate contamination:

1. Each image is resized to 9×8 pixels
2. Per-row horizontal pixel differences yield a 64-bit hash
3. Images with Hamming distance ≤ 8 are grouped into the same cluster
4. Clusters are assigned **entirely to one split** — no cluster spans train and test

```python
# dHash clustering (simplified)
def dhash(image, hash_size=8):
    img = cv2.resize(image, (hash_size + 1, hash_size))
    diff = img[:, 1:] > img[:, :-1]
    return sum([2**i for i, v in enumerate(diff.flatten()) if v])

# Cluster-aware split: entire cluster stays in one partition
for cluster_id, cluster_images in clusters.items():
    assigned_split = assign_to_split(cluster_id, ratios=[0.70, 0.15, 0.15])
    for img in cluster_images:
        split_assignments[img] = assigned_split
```

**Result:** The leakage-aware split on the RGB dataset produced an **honest 94.33% test accuracy** — a far more credible and deployable outcome.

> **Key insight:** A 5.67% accuracy drop from fixing data leakage is far better than discovering a 50% generalization gap in production. Honest evaluation is a feature, not a limitation.

---

## Training Methodology

### Architecture

- **Backbone:** MobileNetV2 (ImageNet pre-trained via `torchvision.models`)
- **Classifier head:** `nn.Linear(1280, 31)` replacing default classifier
- **Parameters:** ~3.5M total

### Two-Phase Fine-Tuning

```
Phase 1 — Head Warm-up (5 epochs)
  ├── Backbone: frozen
  ├── Learning rate: 1e-3
  └── Purpose: initialize classifier without corrupting ImageNet features

Phase 2 — Full Fine-tuning (up to 25 epochs)
  ├── Backbone: unfrozen
  ├── Learning rate: 5e-5
  ├── Weight decay: 1e-4
  ├── Label smoothing: 0.1
  ├── Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
  └── Early stopping: patience=6
```

### Data Augmentation (Training Only)

| Transform | Parameters | Rationale |
|-----------|-----------|-----------|
| RandomRotation | ±10° | Camera tilt variation |
| RandomAffine | 8% translate, 0.9–1.1 scale | Hand position variance |
| ColorJitter | brightness 0.3, contrast 0.3 | Lighting variation |
| GaussNoise | var 5–25 | Sensor noise simulation |
| MotionBlur | 3px kernel | Motion during signing |
| RandomErasing | 2–10% area | Occlusion robustness |
| **No HorizontalFlip** | — | Mirroring changes sign meaning in ArSL |

---

## Evaluation Metrics

All metrics are computed on the held-out test split (15% of dataset, cluster-isolated from training data):

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **94.33%** |
| Macro Precision | 94.59% |
| Macro Recall | 94.33% |
| Macro F1-Score | 94.36% |
| Best Validation Accuracy | 93.32% (epoch 24) |
| Samples evaluated | 1,165 |

### Per-Class Highlights

| Performance Band | Classes |
|-----------------|---------|
| Perfect (100%) | ha, jeem, ghain |
| Excellent (≥97%) | ain, aleff, laam, waw, zay |
| Good (90–96%) | Most remaining classes |
| Challenging (80–89%) | dal, fa, seen, khaa, ta, dha |

> Full per-class breakdown: [classification_report.csv](CODE/arsigntutor_final_results/final_project_package/results_clean_training/rgb_mobilenetv2_training/classification_report.csv)

### Visual Results

| Asset | Description |
|-------|-------------|
| ![Confusion Matrix](assets/confusion_matrix.png) | Normalized 31×31 confusion matrix |
| ![Training Accuracy](assets/training_accuracy.png) | Validation accuracy over 30 epochs |
| ![Training Loss](assets/training_loss.png) | Train/val loss curves |
| ![Prediction Example](assets/prediction_example.png) | Per-class prediction error grid |
| ![App Demo](assets/app_demo.png) | Streamlit interface in action |

---

## Resume Highlights

- Built an Arabic Sign Language recognition system covering all 31 Arabic letter classes using MobileNetV2 and PyTorch transfer learning.
- Achieved **94.33% test accuracy** after discovering and resolving a severe data leakage issue that had artificially inflated results to 99.26%.
- Identified data contamination via cross-dataset evaluation (48.78% accuracy on an independent dataset), then engineered a dHash-based duplicate detection pipeline to produce a clean, leakage-free train/test split.
- Developed a real-time Streamlit web application with live MediaPipe hand tracking, temporal prediction smoothing, and a gamified letter-spelling interface.
- Designed and executed a full ML pipeline: dataset engineering → augmentation → two-phase fine-tuning → evaluation → deployment, with reproducible split files committed to version control.

---

## Repository Structure

```
ArSignTutor/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Deployment dependencies
├── requirements-dev.txt               # Training & development dependencies
│
├── CODE/
│   ├── app.py                         # Streamlit web application (entry point)
│   ├── config.py                      # Configuration constants
│   ├── arsign_retrain_kaggle.py       # Full training pipeline (Kaggle GPU)
│   ├── evaluate_cross_dataset.py      # Cross-dataset evaluation script
│   └── src/
│       ├── model.py                   # ArSignModel (MobileNetV2 wrapper)
│       ├── dataloader.py              # Dataset loading utilities
│       ├── training/
│       │   └── train_model.py         # Local training pipeline
│       ├── evaluation/
│       │   └── evaluate.py            # Test set evaluation
│       └── preprocessing/
│           ├── transforms.py          # Augmentation pipeline
│           ├── split_dataset.py       # Leakage-aware dataset splitting
│           └── dataset_analysis.py    # Class distribution analysis
│
├── CODE/arsigntutor_final_results/
│   ├── best_rgb_domain_adapted_mobilenetv2.pth  # Model weights (see Releases)
│   ├── class_names.json
│   └── final_project_package/results_clean_training/
│       ├── rgb_mobilenetv2_training/
│       │   ├── training_history.json
│       │   ├── classification_report.csv
│       │   └── figures/
│       └── splits/                    # Committed split files for reproducibility
│
├── assets/                            # Visual assets
│   ├── app_demo.png
│   ├── confusion_matrix.png
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── prediction_example.png
│
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   ├── MODEL_TRAINING.md
│   ├── DATASET_PREPARATION.md
│   └── EVALUATION_METHODOLOGY.md
│
└── Report/
    └── ZEED_AL_HAJ_ALI_Report.pdf
```

---

## Installation

### Prerequisites

- Python 3.9+
- Webcam (for real-time inference)
- GPU optional — CPU is sufficient for inference

### Setup

```bash
# Clone the repository
git clone https://github.com/ZEYID-H/ArSignTutor.git
cd ArSignTutor

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install deployment dependencies
pip install -r requirements.txt
```

For training and evaluation:

```bash
pip install -r requirements-dev.txt
```

---

## Usage

### Run the Web Application

```bash
streamlit run CODE/app.py
```

Open `http://localhost:8501` and allow webcam access.

**How to play:**
1. Select difficulty: **Easy** (3-letter words) or **Hard** (4–5 letter words)
2. An Arabic word appears on screen with individual letter boxes
3. Sign each letter in sequence using your hand in front of the webcam
4. The system detects and validates each sign in real time
5. Earn points for correct letters; build streaks for bonus points

### Run Evaluation

```bash
# Evaluate on the held-out test split
python CODE/src/evaluation/evaluate.py

# Cross-dataset generalization test
python CODE/evaluate_cross_dataset.py
```

### Train from Scratch (Kaggle)

Upload `CODE/arsign_retrain_kaggle.py` to a Kaggle notebook with:
- Dataset: RGB Arabic Alphabets Sign Language
- Dataset: ArASL2018 (for cross-dataset validation)
- Accelerator: GPU T4 x1

```bash
# Local training
python CODE/src/training/train_model.py
```

---

## Streamlit Deployment

### Local

```bash
streamlit run CODE/app.py
```

### Streamlit Community Cloud

1. Push this repository to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect the `ZEYID-H/ArSignTutor` repository
4. Set **Main file path:** `CODE/app.py`
5. Deploy

> **Note on model weights:** The `.pth` checkpoint is hosted in GitHub Releases (not tracked by git). Download it from the [Releases](https://github.com/ZEYID-H/ArSignTutor/releases) page and place it at `CODE/arsigntutor_final_results/best_rgb_domain_adapted_mobilenetv2.pth` before running.

---

## Future Improvements

| Priority | Improvement |
|----------|------------|
| High | Auto-download model weights from GitHub Releases on first run |
| High | Expand vocabulary to full MSA word list (500+ words) |
| Medium | Signer-independent evaluation using held-out signer splits |
| Medium | ArSL continuous signing support (word-level, not letter-level) |
| Medium | ONNX export for 2–3× CPU inference speedup |
| Low | Mobile deployment via TensorFlow Lite or ONNX Runtime Mobile |
| Low | Community data collection module for new signer contributions |
| Low | Gulf Arabic Sign Language dialect support |

---

## GitHub Topics

`artificial-intelligence` `machine-learning` `deep-learning` `computer-vision` `pytorch` `streamlit` `mediapipe` `sign-language-recognition` `arabic-sign-language` `dataset-engineering` `model-evaluation` `transfer-learning` `mobilenetv2` `data-leakage`

> To add these to the repository: go to the repository page on GitHub → click the gear icon next to **About** → paste the topics above.

---

## Author

**Zeed Al Haj Ali**

- GitHub: [@ZEYID-H](https://github.com/ZEYID-H)
- Email: zeednehd@gmail.com

---

## Citation

```bibtex
@misc{alhajali2024arsigntutor,
  author = {Al Haj Ali, Zeed},
  title  = {ArSignTutor: Interactive Real-Time Arabic Sign Language Learning System},
  year   = {2024},
  url    = {https://github.com/ZEYID-H/ArSignTutor}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [MediaPipe](https://mediapipe.dev) — hand landmark detection
- [ArASL Dataset](https://www.kaggle.com/datasets/ammarsayedtaha/arabic-sign-language-dataset-2022) — reference dataset for leakage analysis
- [RGB Arabic Alphabets Dataset](https://www.kaggle.com/datasets/monasaleh/arabic-alphabets-sign-language-dataset) — clean training dataset
- [MobileNetV2](https://arxiv.org/abs/1801.04381) — backbone architecture
