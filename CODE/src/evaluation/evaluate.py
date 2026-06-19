import sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import config
from src.model import ArSignModel
from src.dataloader import get_dataloaders

DEVICE = config.DEVICE


def load_best_model(num_classes, fallback_names):
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"No weights at {config.BEST_MODEL_PATH}. Run train_model.py first.")
    ck = torch.load(config.BEST_MODEL_PATH, map_location=DEVICE)
    m  = ArSignModel(num_classes=ck.get("num_classes", num_classes))
    m.load_state_dict(ck["model_state_dict"]); m.to(DEVICE).eval()
    return m, ck.get("class_names", fallback_names)


@torch.no_grad()
def evaluate():
    config.set_seed(config.RANDOM_SEED)
    _, _, te_loader, _, _, te_ds = get_dataloaders()
    model, names = load_best_model(len(te_ds.classes), te_ds.classes)
    criterion = nn.CrossEntropyLoss()

    preds, labels, loss_sum, total = [], [], 0.0, 0
    for imgs, lbl in te_loader:
        imgs, lbl = imgs.to(DEVICE), lbl.to(DEVICE)
        out = model(imgs)
        loss_sum += criterion(out, lbl).item() * imgs.size(0)
        total    += lbl.size(0)
        preds.extend(torch.max(out, 1)[1].cpu().numpy())
        labels.extend(lbl.cpu().numpy())

    preds, labels = np.array(preds), np.array(labels)
    acc = (preds == labels).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    report = classification_report(labels, preds, target_names=names, digits=2, zero_division=0)

    header = (f"FINAL TEST EVALUATION\n{'='*60}\n"
              f"Loss: {loss_sum/total:.4f}  Acc: {acc*100:.2f}%  P: {prec:.4f}  R: {rec:.4f}  F1: {f1:.4f}\n\n")
    full = header + report
    print(full)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "classification_report.txt").write_text(full, encoding="utf-8")

    ticks = np.arange(len(names))
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix(labels, preds), interpolation="nearest", cmap="Blues")
    plt.title("ArSignTutor Confusion Matrix (Test Set)"); plt.colorbar()
    plt.xticks(ticks, names, rotation=90, fontsize=7); plt.yticks(ticks, names, fontsize=7)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=200); plt.close()

    print(f"Report → {config.REPORTS_DIR/'classification_report.txt'}")
    print(f"Figure → {config.FIGURES_DIR/'confusion_matrix.png'}")


if __name__ == "__main__":
    evaluate()
