import sys
from pathlib import Path
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import config
from src.preprocessing.transforms import train_transform


def main():
    if not config.TRAIN_DIR.exists(): print(f"Train dir not found: {config.TRAIN_DIR}"); return
    cls_folders = [f for f in config.TRAIN_DIR.iterdir() if f.is_dir()]
    if not cls_folders: print("No class folders found."); return
    img_path = next(iter(cls_folders[0].glob("*")), None)
    if img_path is None: print("No images found."); return
    img = Image.open(img_path).convert("RGB"); t = train_transform(img)
    print(f"Path: {img_path}\nMode: {img.mode}\nShape: {t.shape}\nMin/Max: {t.min():.4f} / {t.max():.4f}")


if __name__ == "__main__":
    main()
