import sys
from pathlib import Path
import numpy as np
import albumentations as A
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import config


class _AlbumentationsAug:
    def __init__(self, aug): self._aug = aug
    def __call__(self, pil_img): return self._aug(image=np.array(pil_img))["image"]


_aug = A.Compose([
    A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
    A.Rotate(limit=10, border_mode=0, p=0.6),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, border_mode=0, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0, p=0.5),
    A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
])

_norm = transforms.Normalize(mean=config.NORMALIZATION_MEAN, std=config.NORMALIZATION_STD)

train_transform = transforms.Compose([_AlbumentationsAug(_aug), transforms.ToTensor(), _norm])
eval_transform  = transforms.Compose([transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), transforms.ToTensor(), _norm])
