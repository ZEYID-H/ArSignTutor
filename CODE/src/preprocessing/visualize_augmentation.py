import sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import config
from src.preprocessing.transforms import train_transform


def unnormalize(t):
    for ch, m, s in zip(t, config.NORMALIZATION_MEAN, config.NORMALIZATION_STD):
        ch.mul_(s).add_(m)
    return t.clamp(0, 1)


def main():
    if not config.TRAIN_DIR.exists(): print(f"Train dir not found: {config.TRAIN_DIR}"); return
    cls_folders = [f for f in config.TRAIN_DIR.iterdir() if f.is_dir()]
    if not cls_folders: print("No class folders found."); return
    img_path = next(iter(cls_folders[0].glob("*")), None)
    if img_path is None: print("No images found."); return

    img = Image.open(img_path).convert("RGB")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(unnormalize(train_transform(img).clone()).permute(1, 2, 0))
        plt.axis("off"); plt.title(f"Aug {i+1}")
    plt.tight_layout()
    out = config.FIGURES_DIR / "augmentation_examples.png"
    plt.savefig(out, dpi=300); plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
