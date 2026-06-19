# ArSignTutor — Arabic Sign Language Learning System

A university project that teaches Arabic Sign Language through a live webcam app.
The user shows hand signs in front of the camera, and the system recognizes the letter in real time.

---

## What the app does

- You open the app in your browser
- A word appears on screen in Arabic
- You spell it letter by letter using hand signs in front of your webcam
- The system tells you if you got it right and gives you points

---

## The scientific story (important)

The project went through two phases:

**Phase 1 — Old model (ArASL dataset)**
We trained MobileNetV2 on the ArASL dataset and got 99.26% accuracy.
But we discovered this number was likely inflated — the dataset has many near-duplicate images from the same recording session, and a random split lets similar frames appear in both training and testing.

**Phase 2 — Cross-dataset test**
We tested the old model on a completely different dataset (RGB Arabic Alphabets).
Accuracy dropped to 48.78%. This confirmed the old result did not generalize well.

**Phase 3 — Clean retraining (final result)**
We used the RGB Arabic Alphabets dataset and built a leakage-aware split using dHash near-duplicate clustering, so similar images always stay in the same partition.
We retrained MobileNetV2 on this clean split and got **94.33% test accuracy** — a realistic, honest result.

This 94.33% is the main result of the project.

---

## Final model results

| Metric | Value |
|--------|-------|
| Test Accuracy | **94.33%** |
| Precision | 94.59% |
| Recall | 94.33% |
| F1-Score | 94.36% |
| Best Validation Accuracy | 93.32% |
| Classes supported | 31 Arabic letters (yaa excluded) |

---

## How to run the app

Make sure you have Python installed, then:

```bash
pip install streamlit streamlit-webrtc torch torchvision mediapipe opencv-python pillow
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.
Click **START** in the camera area, allow webcam access, and start signing.

---

## Project structure

```
ArSignTutor3/
├── app.py                         # The main app (Streamlit)
├── arsign_retrain_kaggle.py       # Training script (ran on Kaggle GPU)
├── evaluate_cross_dataset.py      # Cross-dataset evaluation script
├── config.py                      # Settings and paths
├── hand_landmarker.task           # MediaPipe hand detection model
├── arsigntutor_final_results/
│   ├── best_rgb_domain_adapted_mobilenetv2.pth   # Final model weights
│   └── class_names.json                          # 31 class names
└── src/
    ├── model.py
    ├── dataloader.py
    ├── training/train_model.py
    └── evaluation/evaluate.py
```

---

## Technology used

| What | Tool |
|------|------|
| Model | MobileNetV2 (PyTorch) |
| Training | Kaggle GPU (Tesla T4) |
| Hand detection | MediaPipe |
| App framework | Streamlit |
| Language | Python |

---

## Limitations

- Only isolated alphabet signs, not full sentences
- The yaa letter is excluded (not in the RGB dataset)
- Model performance may vary with different lighting or camera angles
- No mobile version yet

---

*University Project — Artificial Intelligence Department · 2026*
