# Project Structure

This document describes every file and directory in the ArSignTutor repository and its purpose.

---

## Top-Level Layout

```
ArSignTutor/
├── README.md                    Main project documentation
├── LICENSE                      MIT License
├── .gitignore                   Git exclusion rules
├── requirements.txt             Deployment dependencies (app only)
├── requirements-dev.txt         Training & evaluation dependencies
│
├── CODE/                        All source code
├── assets/                      Visual assets for documentation
├── docs/                        Extended technical documentation
└── Report/                      Academic report (PDF + LaTeX source)
```

---

## CODE/ — Source Code

### Entry Points

| File | Purpose |
|------|---------|
| `CODE/app.py` | Streamlit web application — the main user-facing entry point |
| `CODE/config.py` | Global configuration: paths, hyperparameters, model constants |
| `CODE/arsign_retrain_kaggle.py` | Complete training pipeline designed for Kaggle GPU environment |
| `CODE/evaluate_cross_dataset.py` | Tests a trained model on a different dataset to measure generalization |

### CODE/src/ — Core Library

```
CODE/src/
├── model.py              ArSignModel class — MobileNetV2 wrapper with 31-class head
├── dataloader.py         Dataset loading, class mapping, and DataLoader construction
├── check_model.py        Sanity-check model architecture and forward pass shape
├── check_dataloader.py   Verify dataloader returns correct shapes and labels
│
├── training/
│   └── train_model.py    Local training pipeline (CPU/GPU, with ONNX export)
│
├── evaluation/
│   └── evaluate.py       Evaluate model on held-out test split; output metrics + confusion matrix
│
└── preprocessing/
    ├── transforms.py              Albumentations augmentation pipeline (train/val/test variants)
    ├── split_dataset.py           Leakage-aware cluster-based train/val/test splitting
    ├── dataset_analysis.py        Class distribution statistics and imbalance reporting
    ├── check_dataset.py           Verify dataset directory structure and file integrity
    ├── check_preprocessing.py     Visualize augmentation output on sample images
    ├── check_split.py             Verify split sizes and cluster isolation
    └── visualize_augmentation.py  Grid visualization of augmented samples per class
```

### CODE/src/app/ — Runtime Assets

| File | Purpose |
|------|---------|
| `hand_landmarker.task` | MediaPipe HandLandmarker model binary (required at inference time, ~8 MB) |

### CODE/arsigntutor_final_results/ — Trained Model & Outputs

```
arsigntutor_final_results/
├── best_rgb_domain_adapted_mobilenetv2.pth    Final model weights checkpoint
├── class_names.json                            31-class label mapping
└── final_project_package/
    └── results_clean_training/
        ├── rgb_mobilenetv2_training/
        │   ├── best_rgb_domain_adapted_mobilenetv2.pth   (copy — use root version)
        │   ├── training_history.json                      Per-epoch loss/accuracy
        │   ├── training_history.csv                       Same data in CSV format
        │   ├── final_metrics_summary.csv                  Aggregate test metrics
        │   ├── classification_report.csv                  Per-class precision/recall/F1
        │   ├── figures/                                   Generated plots (confusion matrix, curves)
        │   └── reports/                                   Supplementary analysis outputs
        └── splits/
            ├── rgb_train_files.json            Train split file list
            ├── rgb_val_files.json              Validation split file list
            ├── rgb_test_files.json             Test split file list
            └── rgb_cluster_split_all.csv       Full cluster assignment table
```

---

## assets/ — Visual Assets

Populated by copying figures from the training results or from screenshots:

| File | Source |
|------|--------|
| `confusion_matrix_normalized.png` | `results_clean_training/figures/` |
| `training_curves.png` | `results_clean_training/figures/` |
| `error_grid.png` | `results_clean_training/figures/` |
| `app_screenshot.png` | Screenshot of running Streamlit app |

---

## docs/ — Extended Documentation

| File | Contents |
|------|---------|
| `PROJECT_STRUCTURE.md` | This file |
| `MODEL_TRAINING.md` | Full training pipeline walkthrough and hyperparameter reference |
| `DATASET_PREPARATION.md` | Dataset download, dHash clustering, and split generation instructions |
| `EVALUATION_METHODOLOGY.md` | Evaluation design, metrics definitions, and data leakage case study |

---

## Report/ — Academic Report

| File | Purpose |
|------|---------|
| `Report/ZEED_AL_HAJ_ALI_Report.pdf` | Final academic paper (PDF) |
| `Report/Overleaf_Source/` | LaTeX source for reproduction |

---

## What Is NOT in the Repository

The following items are excluded via `.gitignore` due to size or sensitivity:

| Item | Reason | Where to get it |
|------|--------|----------------|
| Dataset images | Too large (GB-scale) | Kaggle (see DATASET_PREPARATION.md) |
| `.pth` model weights | Large binary files | Included in release assets or request from author |
| `venv/` | Environment-specific | Recreate with `pip install -r requirements.txt` |
| `.env` | May contain secrets | Not used; no secrets required |
