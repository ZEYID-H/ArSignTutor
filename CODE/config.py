import random
from pathlib import Path
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR           = PROJECT_ROOT / "dataset"
RAW_DATASET_DIR       = DATASET_DIR / "raw/archive/ArASL_Database_54K_Final/ArASL_Database_54K_Final"
PROCESSED_DATASET_DIR = DATASET_DIR / "processed_dataset"
TRAIN_DIR  = PROCESSED_DATASET_DIR / "train"
VAL_DIR    = PROCESSED_DATASET_DIR / "val"
TEST_DIR   = PROCESSED_DATASET_DIR / "test"

MODELS_DIR       = PROJECT_ROOT / "models"
RESULTS_DIR      = PROJECT_ROOT / "results"
OUTPUTS_DIR      = PROJECT_ROOT / "outputs"
FIGURES_DIR      = OUTPUTS_DIR / "figures"
REPORTS_DIR      = OUTPUTS_DIR / "reports"
CHECKPOINTS_DIR  = OUTPUTS_DIR / "checkpoints"
BEST_MODEL_PATH  = MODELS_DIR / "best_model.pth"

SRC_DIR           = PROJECT_ROOT / "src"
TRAINING_DIR      = SRC_DIR / "training"
EVALUATION_DIR    = SRC_DIR / "evaluation"
PREPROCESSING_DIR = SRC_DIR / "preprocessing"
WEBCAM_DIR        = SRC_DIR / "webcam"
APP_DIR           = SRC_DIR / "app"
UTILS_DIR         = SRC_DIR / "utils"

RANDOM_SEED = 42
IMAGE_SIZE  = 224
CHANNELS    = 3
NUM_CLASSES = 31

NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD  = [0.229, 0.224, 0.225]

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if _TORCH_AVAILABLE else "cpu"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

BATCH_SIZE      = 32
NUM_EPOCHS      = 15
LEARNING_RATE   = 0.001
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.05
NUM_WORKERS     = 0  # 0 is safe on Windows; raise on Linux for speed

CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW     = 5

APP_TITLE       = "ArSignTutor"
APP_DESCRIPTION = "Interactive Arabic Sign Language Learning System"


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
