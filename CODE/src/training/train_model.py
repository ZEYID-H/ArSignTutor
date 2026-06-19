import sys
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import config
from src.model import ArSignModel
from src.dataloader import get_dataloaders

DEVICE = config.DEVICE


def run_epoch(model, loader, criterion, optimizer=None, desc="Train"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss_sum = correct = total = 0
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for imgs, lbls in tqdm(loader, desc=desc, leave=False):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            if is_train: optimizer.zero_grad()
            out = model(imgs); loss = criterion(out, lbls)
            if is_train: loss.backward(); optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct  += (torch.max(out, 1)[1] == lbls).sum().item()
            total    += lbls.size(0)
    return loss_sum / total, correct / total


def main():
    config.set_seed(config.RANDOM_SEED)
    print(f"{'='*60}\nARSIGNTUTOR TRAINING  |  device: {DEVICE}\n{'='*60}")

    tr_loader, va_loader, _, tr_ds, va_ds, _ = get_dataloaders()
    names = tr_ds.classes; n = len(names)
    print(f"Train: {len(tr_ds)}  Val: {len(va_ds)}  Classes: {n}")

    model     = ArSignModel(num_classes=n).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = config.OUTPUTS_DIR / "tensorboard_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_acc = 0.0
    for ep in range(config.NUM_EPOCHS):
        tr_loss, tr_acc = run_epoch(model, tr_loader, criterion, optimizer, "Training")
        va_loss, va_acc = run_epoch(model, va_loader, criterion, desc="Validation")
        scheduler.step(va_acc)
        print(f"Ep {ep+1:2d}/{config.NUM_EPOCHS}  train {tr_loss:.4f}/{tr_acc*100:.2f}%  val {va_loss:.4f}/{va_acc*100:.2f}%")
        for tag, val in [("Loss/train",tr_loss),("Loss/val",va_loss),("Acc/train",tr_acc),("Acc/val",va_acc)]:
            writer.add_scalar(tag, val, ep+1)
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model_state_dict":model.state_dict(),"class_names":names,
                        "num_classes":n,"best_val_acc":best_acc,"epoch":ep+1}, config.BEST_MODEL_PATH)
            print(f"  → saved (val acc {best_acc*100:.2f}%)")
    writer.close()
    print(f"\nBest val acc: {best_acc*100:.2f}%  |  {config.BEST_MODEL_PATH}")

    ckpt = torch.load(config.BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    dummy    = torch.randn(1, config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE).to(DEVICE)
    onnx_path = config.MODELS_DIR / "best_model.onnx"
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image":{0:"batch"},"logits":{0:"batch"}}, opset_version=17)
    print(f"ONNX exported → {onnx_path}")


if __name__ == "__main__":
    main()
