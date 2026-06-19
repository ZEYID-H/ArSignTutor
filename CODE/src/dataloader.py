import sys
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import config
from src.preprocessing.transforms import train_transform, eval_transform


def get_dataloaders():
    tr = ImageFolder(root=config.TRAIN_DIR, transform=train_transform)
    va = ImageFolder(root=config.VAL_DIR,   transform=eval_transform)
    te = ImageFolder(root=config.TEST_DIR,  transform=eval_transform)
    kw = dict(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    return (DataLoader(tr, shuffle=True, **kw), DataLoader(va, **kw), DataLoader(te, **kw), tr, va, te)

def get_class_names():
    return ImageFolder(root=config.TRAIN_DIR, transform=eval_transform).classes
