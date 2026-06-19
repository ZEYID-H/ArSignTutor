# evaluate_cross_dataset.py
# Tests the trained model on a different dataset to measure real-world generalization.
#
# Usage:
#   python evaluate_cross_dataset.py --new-data /path/to/dataset
#   python evaluate_cross_dataset.py --new-data /path/to/dataset --per-class 50

import argparse, json, random, sys
from pathlib import Path

import cv2, numpy as np, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config

CLASS_NAMES_JSON = ROOT / "arsigntutor_training_results" / "class_names.json"
MODEL_PATH       = ROOT / "models" / "best_model.pth"
RESULTS_DIR      = ROOT / "results_cross"
IN_DIST_ACC      = 0.9926   # original in-distribution accuracy (ArASL, random split)

# Preprocessing: CLAHE grayscale conversion to match original training pipeline
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def to_arasl(pil_img):
    a    = np.array(pil_img)
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a
    return Image.fromarray(cv2.cvtColor(_CLAHE.apply(gray), cv2.COLOR_GRAY2RGB))

TRANSFORM = transforms.Compose([
    transforms.Lambda(to_arasl),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(device, num_classes):
    ck = torch.load(MODEL_PATH, map_location=device)
    m  = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(1280, num_classes)
    m.load_state_dict(ck["model_state_dict"])
    return m.to(device).eval()


_ALIASES = {
    "alef":"aleff","alif":"aleff","elif":"aleff",
    "zain":"zay","zein":"zay",
    "taa_marbuta":"toot","ta_marbuta":"toot",
    "ا":"aleff","ب":"bb","ت":"taa","ث":"thaa","ج":"jeem","ح":"haa",
    "خ":"khaa","د":"dal","ذ":"thal","ر":"ra","ز":"zay","س":"seen",
    "ش":"sheen","ص":"saad","ض":"dhad","ط":"ta","ظ":"dha","ع":"ain",
    "غ":"ghain","ف":"fa","ق":"gaaf","ك":"kaaf","ل":"laam","م":"meem",
    "ن":"nun","ه":"ha","و":"waw","ي":"ya","ة":"toot","ى":"yaa",
    "لا":"la","ال":"al",
}

def build_mapping(new_folders, train_classes):
    idx = {c: i for i, c in enumerate(train_classes)}
    mapping, unmatched = {}, []
    for folder in new_folders:
        key = folder.lower()
        if key in idx:
            mapping[folder] = idx[key]
        elif key in _ALIASES and _ALIASES[key] in idx:
            mapping[folder] = idx[_ALIASES[key]]
            print(f"  alias: '{folder}' -> '{_ALIASES[key]}'")
        else:
            unmatched.append(folder)
    if unmatched:
        print(f"  Not matched ({len(unmatched)}): {unmatched}")
    missing = [c for c in train_classes if c not in {train_classes[v] for v in mapping.values()}]
    if missing:
        print(f"  Missing from dataset: {missing}")
    print(f"  Matched: {len(mapping)}/{len(train_classes)} classes")
    return mapping


def subsample(dataset, per_class):
    by_class = {}
    for i, (_, lbl) in enumerate(dataset.samples):
        by_class.setdefault(lbl, []).append(i)
    indices = []
    for idxs in by_class.values():
        random.shuffle(idxs); indices.extend(idxs[:per_class])
    return Subset(dataset, indices)


def run_inference(model, loader, device, new_idx_to_train_idx):
    preds, labels, positions = [], [], []
    pos = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            out = torch.argmax(model(imgs.to(device)), dim=1).cpu().numpy()
            for j, (p, l) in enumerate(zip(out, lbls.numpy())):
                if l in new_idx_to_train_idx:
                    preds.append(p)
                    labels.append(new_idx_to_train_idx[l])
                    positions.append(pos + j)
            pos += len(lbls)
    return np.array(preds), np.array(labels), positions


def save_results(preds, labels, active_train_idx, active_names, tag):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_correct = int((preds == labels).sum())
    acc = n_correct / max(len(labels), 1)

    report = classification_report(
        labels, preds, labels=active_train_idx,
        target_names=active_names, digits=3, zero_division=0
    )
    rpath = RESULTS_DIR / f"{tag}_report.txt"
    rpath.write_text(report, encoding="utf-8")

    cm  = confusion_matrix(labels, preds, labels=active_train_idx)
    row = cm.sum(axis=1, keepdims=True)
    cmn = cm.astype(float) / np.where(row == 0, 1, row)

    _, axes = plt.subplots(1, 2, figsize=(22, 10))
    for ax, data, title in zip(axes, [cm, cmn], ["Raw counts", "Row-normalized"]):
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_title(f"{tag}  ({acc*100:.2f}%)  -  {title}", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        t = np.arange(len(active_names))
        ax.set_xticks(t); ax.set_yticks(t)
        ax.set_xticklabels(active_names, rotation=90, fontsize=6)
        ax.set_yticklabels(active_names, fontsize=6)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    cmpath = RESULTS_DIR / f"{tag}_confusion_matrix.png"
    plt.savefig(cmpath, dpi=150); plt.close()

    per_class = []
    for ti, name in zip(active_train_idx, active_names):
        mask = labels == ti
        if mask.sum() == 0:
            per_class.append(f"  {name:<12} -  no samples")
        else:
            per_class.append(f"  {name:<12} {(preds[mask]==labels[mask]).mean()*100:6.2f}%  ({mask.sum()})")

    print(f"\n{tag.upper()}")
    print(f"  Accuracy: {acc*100:.2f}%  ({n_correct}/{len(labels)})")
    print(f"  Report: {rpath}")
    print("\nPer-class:\n" + "\n".join(per_class))
    print(f"\n{report}")
    return acc


def save_error_grid(dataset, positions, preds, labels, train_idx_to_name, tag, n=40):
    wrong = [(positions[i], labels[i], preds[i])
             for i in range(len(preds)) if preds[i] != labels[i]]
    if not wrong:
        print("  No errors - 100% accuracy!"); return
    random.shuffle(wrong); wrong = wrong[:n]
    cols = 8; rows = (len(wrong) + cols - 1) // cols
    _, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    axes = np.array(axes).flatten()
    for ax in axes: ax.axis("off")
    _mean = np.array([0.485, 0.456, 0.406]); _std = np.array([0.229, 0.224, 0.225])
    for ax, (di, tl, pl) in zip(axes, wrong):
        img, _ = dataset[di]
        ax.imshow((img.permute(1,2,0).numpy() * _std + _mean).clip(0,1))
        ax.axis("off")
        ax.set_title(f"T:{train_idx_to_name.get(tl,tl)}\nP:{train_idx_to_name.get(pl,pl)}", fontsize=6)
    plt.tight_layout()
    path = RESULTS_DIR / f"{tag}_errors.png"
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Error grid ({len(wrong)} samples) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-data",    required=True, type=Path,
                    help="Root folder of the new dataset (one sub-folder per class)")
    ap.add_argument("--per-class",   type=int, default=0,
                    help="Max images per class (0 = use all)")
    ap.add_argument("--sanity-n",    type=int, default=10,
                    help="Images per class for sanity check")
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--batch-size",  type=int, default=32)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nModel: {MODEL_PATH}\n")

    with open(CLASS_NAMES_JSON) as f:
        train_classes = json.load(f)
    n_cls = len(train_classes)
    print(f"Training classes ({n_cls}): {train_classes}\n")

    model = load_model(device, n_cls)

    # Sanity check on the original test split
    sanity_acc = None
    if not args.skip_sanity:
        print(f"Sanity check - original ArASL test split ({args.sanity_n} imgs/class)")
        if not config.TEST_DIR.exists():
            print(f"  Test dir not found: {config.TEST_DIR}")
            print("  Use --skip-sanity to bypass.\n")
        else:
            s_ds  = ImageFolder(root=config.TEST_DIR, transform=TRANSFORM)
            s_sub = subsample(s_ds, args.sanity_n)
            s_map = {s_ds.class_to_idx[c]: train_classes.index(c)
                     for c in s_ds.classes if c in train_classes}
            s_preds, s_labels, _ = run_inference(
                model, DataLoader(s_sub, batch_size=args.batch_size, shuffle=False, num_workers=0),
                device, s_map
            )
            sanity_acc = (s_preds == s_labels).mean() if len(s_labels) else 0.0
            print(f"  Samples: {len(s_labels)}")
            print(f"  Accuracy: {sanity_acc*100:.2f}% (expected >= 99%)")
            if sanity_acc < 0.90:
                print("  Pipeline broken - stopping.")
                sys.exit(1)
            print("  OK\n")

    # Cross-dataset evaluation
    print("Cross-dataset evaluation")
    if not args.new_data.exists():
        print(f"[ERROR] Path not found: {args.new_data}"); sys.exit(1)

    new_ds = ImageFolder(root=args.new_data, transform=TRANSFORM)
    print(f"Root: {args.new_data}")
    print(f"Folders ({len(new_ds.classes)}): {new_ds.classes}\n")

    mapping = build_mapping(new_ds.classes, train_classes)
    if not mapping:
        print("[ERROR] No classes matched."); sys.exit(1)

    new_idx_to_train = {new_ds.class_to_idx[f]: ti for f, ti in mapping.items()}
    active_train_idx = sorted(set(new_idx_to_train.values()))
    active_names     = [train_classes[i] for i in active_train_idx]
    name_map         = {i: train_classes[i] for i in range(n_cls)}

    eval_ds = subsample(new_ds, args.per_class) if args.per_class > 0 else new_ds
    print(f"Total eval samples: {len(eval_ds)}")

    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    preds, labels, ds_pos = run_inference(model, loader, device, new_idx_to_train)

    if len(labels) == 0:
        print("[ERROR] No samples passed class filtering."); sys.exit(1)

    cross_acc = save_results(preds, labels, active_train_idx, active_names, "cross_dataset")
    save_error_grid(eval_ds, ds_pos, preds, labels, name_map, "cross_dataset")

    gap = (IN_DIST_ACC - cross_acc) * 100
    print(f"\nSummary:")
    if sanity_acc is not None:
        print(f"  Sanity (ArASL test): {sanity_acc*100:.2f}%")
    print(f"  In-distribution (ArASL, random split): {IN_DIST_ACC*100:.2f}%")
    print(f"  Cross-dataset ({args.new_data.name}): {cross_acc*100:.2f}%")
    print(f"  Gap: {gap:+.2f} pp")

    if gap > 20:
        print(f"\n  {gap:.1f}pp gap - confirms data leakage in the original split.")
        print(f"  Cross-dataset {cross_acc*100:.2f}% is the honest estimate.")
    elif gap > 5:
        print(f"\n  Moderate gap ({gap:.1f}pp) - mix of leakage and domain shift.")
    else:
        print(f"\n  Small gap ({gap:.1f}pp) - model generalizes well.")

    print(f"\nResults saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
