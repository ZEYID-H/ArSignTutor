import sys
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import config
from src.model import ArSignModel


def main():
    model = ArSignModel(num_classes=config.NUM_CLASSES)
    dummy = torch.randn(config.BATCH_SIZE, config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out   = model(dummy)
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Input: {dummy.shape}  Output: {out.shape}")
    print(f"Params: {total:,} total / {train:,} trainable")


if __name__ == "__main__":
    main()
