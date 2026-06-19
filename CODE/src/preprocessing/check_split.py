import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import config


def count_images(folder):
    if not folder.exists(): return 0
    return sum(len(list(d.glob("*"))) for d in folder.iterdir() if d.is_dir())


tr = count_images(config.TRAIN_DIR)
va = count_images(config.VAL_DIR)
te = count_images(config.TEST_DIR)
print(f"Train: {tr}  Val: {va}  Test: {te}  Total: {tr+va+te}")
