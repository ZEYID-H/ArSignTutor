# Pre-Publishing Checklist

Complete these steps before pushing to GitHub.

---

## 1. Strip Identifying Information

- [x] ~~Update `README.md` with GitHub username~~ — **Done:** all links use `ZEYID-H`
- [ ] Delete `README_SUBMISSION.txt` (replaced by README.md)
- [ ] Search for any remaining student ID references in source code

```bash
grep -r "2280205" CODE/
grep -r "zeedp" CODE/
```

- [ ] Check `CODE/arsign_retrain_kaggle.py` for any hardcoded Kaggle paths that reference your personal account

```bash
grep -r "2280205" CODE/
grep -r "zeedp" CODE/
```

---

## 2. Add Visual Assets

- [ ] Copy training figures to `assets/`:
  ```
  CODE/arsigntutor_final_results/final_project_package/results_clean_training/
  rgb_mobilenetv2_training/figures/*.png  →  assets/
  ```
- [ ] Rename files to match README references:
  - `confusion_matrix_normalized.png`
  - `training_curves.png`
  - `error_grid.png`
- [ ] Take a screenshot of the running Streamlit app → `assets/app_screenshot.png`

---

## 3. Handle the Model Weights

The trained model (`best_rgb_domain_adapted_mobilenetv2.pth`) is too large for GitHub (>100 MB limit).

**Option A — GitHub Releases (recommended):**
1. Create a GitHub release (v1.0)
2. Attach the `.pth` file as a release asset
3. Add a download note in the README

**Option B — Hugging Face Hub:**
```bash
pip install huggingface-hub
huggingface-cli upload YOUR_HF_USERNAME/arsigntutor \
    CODE/arsigntutor_final_results/best_rgb_domain_adapted_mobilenetv2.pth \
    best_rgb_domain_adapted_mobilenetv2.pth
```
Then update `CODE/app.py` to download from Hub if local file not found.

**Option C — Google Drive:**
Upload and add a direct download link in README.

---

## 4. Verify .gitignore is Working

```bash
git status
# Confirm these do NOT appear as tracked files:
# - Any .pth or .pt files
# - The dataset directory
# - venv/ or env/
# - __pycache__/
```

---

## 5. Remove Redundant Files

The following are safe to delete before the first commit:

- [ ] `README_SUBMISSION.txt` — replaced by `README.md`
- [ ] Duplicate `.pth` file: `CODE/arsigntutor_final_results/final_project_package/results_clean_training/rgb_mobilenetv2_training/best_rgb_domain_adapted_mobilenetv2.pth` — keep only the one in `CODE/arsigntutor_final_results/`
- [ ] `CODE/requirements.txt` — superseded by root-level `requirements.txt` and `requirements-dev.txt`

---

## 6. Final README Review

- [ ] Update GitHub badge URLs to use your real username
- [ ] Verify all internal links work (docs/, assets/, CODE/ paths)
- [ ] Add a real Streamlit Cloud or Hugging Face Spaces demo link if deploying
- [ ] Replace placeholder `[Confusion Matrix](assets/confusion_matrix_normalized.png)` links once images are in place

---

## 7. Repository Settings on GitHub

After pushing:

- [ ] Add topics/tags: `arabic-sign-language`, `deep-learning`, `computer-vision`, `mobilenetv2`, `streamlit`, `mediapipe`, `pytorch`, `transfer-learning`
- [ ] Set repository description: "Real-time Arabic Sign Language learning system using MobileNetV2, MediaPipe, and Streamlit. Includes a data leakage case study (99.26% → 94.33%)."
- [ ] Pin the repository to your profile
- [ ] Enable GitHub Pages if you want to host the academic report

---

## 8. Portfolio Note

The leakage case study (99.26% → 48.78% → 94.33%) is the strongest differentiator — lead with it for ML Engineer and AI Evaluation roles.
