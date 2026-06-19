import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
from src.dataloader import get_dataloaders, get_class_names


def main():
    tr, va, te, trd, vad, ted = get_dataloaders()
    print(f"Train: {len(trd)}  Val: {len(vad)}  Test: {len(ted)}")
    names = get_class_names()
    print(f"Classes ({len(names)}): {names}")
    imgs, lbls = next(iter(tr))
    print(f"Batch: images={imgs.shape}  labels={lbls.shape}  dtype={imgs.dtype}")
    print(f"Labels: {lbls}")


if __name__ == "__main__":
    main()
