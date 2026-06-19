# Model Training Guide

Complete reference for training ArSignTutor's MobileNetV2 classifier on the RGB Arabic Alphabets Sign Language dataset.

---

## Architecture Overview

### Backbone: MobileNetV2

MobileNetV2 was selected for three reasons:

1. **Real-time inference on CPU** — 3.5M parameters, ~22ms per frame on a modern laptop CPU
2. **Transfer learning effectiveness** — ImageNet pre-training provides robust low-level visual features that transfer well to hand gesture recognition
3. **Deployment flexibility** — small enough for browser-based deployment, exportable to ONNX/TFLite

### Model Definition

```python
# CODE/src/model.py
import torch.nn as nn
from torchvision.models import mobilenet_v2

class ArSignModel(nn.Module):
    def __init__(self, num_classes=31):
        super().__init__()
        backbone = mobilenet_v2(weights='IMAGENET1K_V1')
        # Replace 1000-class ImageNet head with 31-class ArSL head
        backbone.classifier[1] = nn.Linear(1280, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)
```

**Input:** 224×224 RGB image tensor, normalized with ImageNet statistics  
**Output:** logits of shape `(batch_size, 31)`

---

## Two-Phase Fine-Tuning Strategy

Training uses a two-phase approach to prevent the "catastrophic forgetting" that occurs when a pre-trained backbone is immediately fine-tuned at a high learning rate.

### Phase 1: Head Warm-up (5 epochs)

```python
# Freeze backbone; only train the new classifier head
for param in model.model.features.parameters():
    param.requires_grad = False

optimizer = AdamW(model.model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
```

**Purpose:** Initialize the 31-class head with meaningful gradients before the backbone starts adapting. Starting full fine-tuning from a random head causes large gradient updates that can destroy learned ImageNet features.

**Expected outcome:** Validation accuracy climbs from ~3% (random) to ~65–70% by epoch 5.

### Phase 2: Full Fine-tuning (up to 25 epochs)

```python
# Unfreeze entire model; use lower LR to preserve backbone features
for param in model.parameters():
    param.requires_grad = True

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
```

**Purpose:** Allow backbone to adapt to the ArSL domain (skin tones, hand textures, lighting) while preserving low-level feature detectors.

**Expected outcome:** Validation accuracy climbs from ~67% to ~93% by epoch 24.

---

## Hyperparameter Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image size | 224×224 | Required by MobileNetV2 |
| Batch size | 64 (Kaggle) / 32 (local) | Larger batch = more stable gradients |
| Phase 1 epochs | 5 | Fixed; not subject to early stopping |
| Phase 2 epochs | up to 25 | Subject to early stopping |
| Phase 1 LR | 1e-3 | Head only |
| Phase 2 LR | 5e-5 | Full model |
| Weight decay | 1e-4 | Applied to all parameters |
| Label smoothing | 0.1 | Prevents overconfident predictions |
| LR factor | 0.5 | Halve LR when val accuracy plateaus |
| LR patience | 3 | Epochs before LR reduction |
| Early stop patience | 6 | Epochs without improvement before stopping |
| Optimizer | AdamW | Adam + decoupled weight decay |
| Loss function | CrossEntropyLoss (label_smoothing=0.1) | |
| Random seed | 42 | Reproducibility |

---

## Data Augmentation Pipeline

Augmentation is applied **only during training**. Validation and test use only resize + normalize.

```python
# CODE/src/preprocessing/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(224, 224),
    A.Rotate(limit=10, p=0.5),                          # ±10° camera tilt
    A.Affine(translate_percent=0.08, scale=(0.9, 1.1)),  # position & scale variance
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
    A.GaussNoise(var_limit=(5, 25), p=0.3),              # sensor noise
    A.MotionBlur(blur_limit=3, p=0.3),                   # motion during signing
    A.CoarseDropout(max_holes=1, max_height=0.1, max_width=0.1, p=0.3),  # occlusion
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# NOTE: HorizontalFlip is intentionally excluded.
# Mirroring a hand sign changes its meaning in Arabic Sign Language.
```

---

## Training on Kaggle (Recommended)

The primary training script `CODE/arsign_retrain_kaggle.py` is designed for Kaggle's free GPU environment (Tesla T4, 16 GB VRAM).

### Setup

1. Create a new Kaggle notebook
2. Add datasets:
   - `monasaleh/arabic-alphabets-sign-language-dataset` (RGB training data)
   - `ammarsayedtaha/arabic-sign-language-dataset-2022` (ArASL — for cross-dataset evaluation)
3. Enable GPU: Settings → Accelerator → GPU T4 x1
4. Upload `CODE/arsign_retrain_kaggle.py` and run

### Expected Runtime

| Phase | Duration |
|-------|---------|
| Data loading & clustering | ~3 minutes |
| Phase 1 (5 epochs) | ~8 minutes |
| Phase 2 (up to 25 epochs) | ~35 minutes |
| Evaluation & plot generation | ~5 minutes |
| **Total** | **~50 minutes** |

### Outputs

The script generates:

```
/kaggle/working/arsigntutor_results/
├── best_rgb_domain_adapted_mobilenetv2.pth    Model checkpoint
├── class_names.json
├── training_history.json / .csv
├── final_metrics_summary.csv
├── classification_report.csv
└── figures/
    ├── training_loss_accuracy.png
    ├── confusion_matrix_raw.png
    ├── confusion_matrix_normalized.png
    └── error_grid_top40.png
```

---

## Local Training

For local training without Kaggle:

```bash
# Install training dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Configure paths in config.py, then:
python CODE/src/training/train_model.py
```

Local training also exports an ONNX model for 2–3× CPU inference speedup:

```python
# Automatic ONNX export at end of training
torch.onnx.export(model, dummy_input, "model.onnx", ...)
```

---

## Monitoring Training

TensorBoard logs are written to `runs/` during local training:

```bash
tensorboard --logdir=runs
# Open http://localhost:6006
```

Key metrics to watch:
- `Loss/train` and `Loss/val` should converge together (divergence = overfitting)
- `Accuracy/val` should increase; plateau triggers LR reduction
- `LearningRate` — look for the LR halving events from ReduceLROnPlateau

---

## Reproducibility

All experiments use seed 42:

```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

The dataset split files (`CODE/arsigntutor_final_results/.../splits/`) are committed to the repository, so evaluation on the exact same test set is always possible without re-running the split.

---

## Loading the Trained Model

```python
import torch
from CODE.src.model import ArSignModel

checkpoint = torch.load('CODE/arsigntutor_final_results/best_rgb_domain_adapted_mobilenetv2.pth',
                        map_location='cpu')
model = ArSignModel(num_classes=checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model trained to {checkpoint['best_val_acc']:.2%} validation accuracy")
print(f"Classes: {checkpoint['class_names']}")
```
