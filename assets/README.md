# assets/

Visual assets for the README and documentation. All files in this directory are tracked by git.

## Required Files

| Filename | Source | Description |
|----------|--------|-------------|
| `confusion_matrix.png` | `CODE/arsigntutor_final_results/.../figures/` | Normalized 31×31 confusion matrix |
| `training_accuracy.png` | `CODE/arsigntutor_final_results/.../figures/` | Validation accuracy curve over epochs |
| `training_loss.png` | `CODE/arsigntutor_final_results/.../figures/` | Train/val loss curves |
| `prediction_example.png` | `CODE/arsigntutor_final_results/.../figures/` | Per-class error grid (top-40 misclassifications) |
| `app_demo.png` | Screenshot of running app | Streamlit interface during a signing session |

## How to Populate

```bash
# Copy training figures (rename to match the filenames above)
cp "CODE/arsigntutor_final_results/final_project_package/results_clean_training/rgb_mobilenetv2_training/figures/confusion_matrix_normalized.png" assets/confusion_matrix.png
cp "CODE/arsigntutor_final_results/final_project_package/results_clean_training/rgb_mobilenetv2_training/figures/training_loss_accuracy.png" assets/training_accuracy.png
cp "CODE/arsigntutor_final_results/final_project_package/results_clean_training/rgb_mobilenetv2_training/figures/error_grid_top40.png" assets/prediction_example.png
```

For `training_loss.png`, use the same training figure or export the loss-only subplot separately.

For `app_demo.png`, run `streamlit run CODE/app.py`, start a signing session, and take a browser screenshot.

## Note

There is also a `training_results.png` in `Report/Overleaf_Source/` — this may be usable as `training_accuracy.png` or `training_loss.png` directly.
