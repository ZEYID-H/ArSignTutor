import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import config

if not config.RAW_DATASET_DIR.exists():
    print(f"Dataset folder not found: {config.RAW_DATASET_DIR}"); raise SystemExit

folders = [f for f in config.RAW_DATASET_DIR.iterdir() if f.is_dir()]
total = 0
for f in sorted(folders):
    n = len(list(f.glob("*"))); total += n; print(f"{f.name}: {n}")
print(f"\nClasses: {len(folders)}  Total images: {total}")
