import random, shutil, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def main():
    config.set_seed(config.RANDOM_SEED)
    if not config.RAW_DATASET_DIR.exists():
        print(f"Raw dataset not found: {config.RAW_DATASET_DIR}"); return
    for d in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    total_tr = total_va = total_te = 0
    for cls in sorted(f for f in config.RAW_DATASET_DIR.iterdir() if f.is_dir()):
        imgs = [p for p in cls.glob("*") if p.suffix.lower() in IMAGE_EXTS]
        random.shuffle(imgs)
        n_tr = int(len(imgs) * config.TRAIN_RATIO)
        n_va = int(len(imgs) * config.VAL_RATIO)
        splits = [(config.TRAIN_DIR, imgs[:n_tr]),
                  (config.VAL_DIR,   imgs[n_tr:n_tr+n_va]),
                  (config.TEST_DIR,  imgs[n_tr+n_va:])]
        for split_dir, batch in splits:
            d = split_dir / cls.name; d.mkdir(parents=True, exist_ok=True)
            for p in batch: shutil.copy(p, d / p.name)
        tr, va, te = len(splits[0][1]), len(splits[1][1]), len(splits[2][1])
        print(f"{cls.name:>8}  train={tr:<5} val={va:<5} test={te:<5}")
        total_tr += tr; total_va += va; total_te += te

    print(f"\nTrain={total_tr}  Val={total_va}  Test={total_te}  Total={total_tr+total_va+total_te}")


if __name__ == "__main__":
    main()
