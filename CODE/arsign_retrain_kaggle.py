# Retraining script for ArSignTutor on Kaggle (GPU)
# Uses the RGB Arabic Alphabets dataset with a leakage-aware split
#
# Before running:
#   1. Add the RGB dataset as Kaggle input
#   2. Add ArASL2018 as Kaggle input
#   3. Enable GPU (T4 or P100)
#   4. Update NEW_DATA_SLUG and ARASL_SLUG below

import os, json, random, time, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

try:
    import imagehash
except ImportError:
    os.system("pip install -q imagehash")
    import imagehash

# Dataset slugs - update to match your Kaggle input names
NEW_DATA_SLUG = "rgb-arabic-alphabets-sign-language-dataset"
ARASL_SLUG    = "arabic-alphabets-sign-language-dataset"

NEW_DATA_ROOT = Path(f"/kaggle/input/{NEW_DATA_SLUG}")
ARASL_ROOT    = Path(f"/kaggle/input/{ARASL_SLUG}")
WORK_DIR      = Path("/kaggle/working")
SPLIT_DIR     = WORK_DIR / "splits"
RESULTS_DIR   = WORK_DIR / "results"
MODEL_OUT     = WORK_DIR / "best_model.pth"
HISTORY_OUT   = WORK_DIR / "training_history.json"

for d in [SPLIT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Fix seeds for reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"Memory  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"PyTorch : {torch.__version__}")

assert NEW_DATA_ROOT.exists(), f"Dataset not found: {NEW_DATA_ROOT}"
assert ARASL_ROOT.exists(),    f"ArASL2018 not found: {ARASL_ROOT}"

# Class names - must match class_names.json order
TRAIN_CLASSES = [
    "ain","al","aleff","bb","dal","dha","dhad","fa","gaaf","ghain",
    "ha","haa","jeem","kaaf","khaa","la","laam","meem","nun","ra",
    "saad","seen","sheen","ta","taa","thaa","thal","toot","waw","ya",
    "yaa","zay"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(TRAIN_CLASSES)}
N_CLASSES    = len(TRAIN_CLASSES)

# Aliases for different folder naming conventions across datasets
_ALIASES = {
    "alef":"aleff","alif":"aleff","elif":"aleff",
    "zain":"zay","zein":"zay",
    "taa_marbuta":"toot","ta_marbuta":"toot","taa marbuta":"toot",
    "ا":"aleff","ب":"bb","ت":"taa","ث":"thaa","ج":"jeem","ح":"haa",
    "خ":"khaa","د":"dal","ذ":"thal","ر":"ra","ز":"zay","س":"seen",
    "ش":"sheen","ص":"saad","ض":"dhad","ط":"ta","ظ":"dha","ع":"ain",
    "غ":"ghain","ف":"fa","ق":"gaaf","ك":"kaaf","ل":"laam","م":"meem",
    "ن":"nun","ه":"ha","و":"waw","ي":"ya","ة":"toot","ى":"yaa",
    "لا":"la","ال":"al",
}
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def resolve_class(folder_name):
    key = folder_name.lower().strip()
    if key in CLASS_TO_IDX: return key
    if key in _ALIASES:     return _ALIASES[key]
    return None

# Scan the new dataset folders
print("\nScanning dataset folders:")
new_class_map = {}
unmatched_new = []
total_images  = 0

for folder in sorted(NEW_DATA_ROOT.iterdir()):
    if not folder.is_dir(): continue
    images = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]
    cls    = resolve_class(folder.name)
    print(f"  {folder.name:<20} {len(images):>5} images  {'-> ' + cls if cls else '!! no match'}")
    if cls:
        new_class_map[folder.name] = cls
        total_images += len(images)
    else:
        unmatched_new.append(folder.name)

print(f"\nTotal: {total_images} images, {len(new_class_map)} classes matched")

sample_folder = next(iter(new_class_map))
sample_imgs   = [p for p in (NEW_DATA_ROOT/sample_folder).rglob("*") if p.suffix.lower() in IMG_EXTS]
sample_arr    = np.array(Image.open(sample_imgs[0]).convert("RGB"))
print(f"Sample image size: {Image.open(sample_imgs[0]).size}, dtype: {sample_arr.dtype}, range: [{sample_arr.min()}, {sample_arr.max()}]")

if unmatched_new:
    print(f"\nUnmatched folders (excluded): {unmatched_new}")

missing = [c for c in TRAIN_CLASSES if c not in new_class_map.values()]
if missing:
    print(f"Classes not found in dataset: {missing}")

assert len(new_class_map) >= N_CLASSES * 0.8, f"Too few classes matched: {len(new_class_map)}/{N_CLASSES}"


# Cluster-based split to avoid data leakage
# Near-duplicate images from the same recording session are grouped together
# using perceptual hashing (dHash), then each group is assigned to one split only.

HASH_THRESHOLD = 8
HASH_WINDOW    = 100
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15

def compute_dhash(path, size=8):
    try:
        return imagehash.dhash(Image.open(path).convert("L"), hash_size=size)
    except Exception:
        return None

def union_find_clusters(items, edges):
    parent = list(range(len(items)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in edges:
        pi, pj = find(i), find(j)
        if pi != pj: parent[pi] = pj
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(items[i])
    return list(groups.values())

def cluster_class_folder(class_folder, threshold=HASH_THRESHOLD, window=HASH_WINDOW):
    # If the dataset has signer sub-folders, use them directly
    signer_dirs = [d for d in class_folder.iterdir() if d.is_dir()]
    if len(signer_dirs) >= 2:
        clusters = []
        for sd in signer_dirs:
            imgs = [p for p in sd.rglob("*") if p.suffix.lower() in IMG_EXTS]
            if imgs: clusters.append(imgs)
        loose = [p for p in class_folder.glob("*") if p.suffix.lower() in IMG_EXTS]
        if loose: clusters.append(loose)
        return clusters

    # Flat folder - cluster by perceptual hash similarity
    all_imgs = [p for p in class_folder.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not all_imgs: return []
    hashes = [(int(str(compute_dhash(p)), 16) if compute_dhash(p) is not None else random.getrandbits(64), p)
              for p in all_imgs]
    hashes.sort(key=lambda x: x[0])
    sorted_imgs = [p for _, p in hashes]
    hash_vals   = [h for h, _ in hashes]

    edges = []
    for i in range(len(hash_vals)):
        for j in range(i+1, min(i+window+1, len(hash_vals))):
            bits = bin(hash_vals[i] ^ hash_vals[j]).count("1")
            if bits <= threshold:
                edges.append((i, j))
            elif hash_vals[j] - hash_vals[i] > (1 << 20):
                break
    return union_find_clusters(sorted_imgs, edges)

def cluster_split(clusters, train_r=TRAIN_RATIO, val_r=VAL_RATIO, seed=SEED):
    rng = random.Random(seed)
    clusters_sorted = sorted(clusters, key=len, reverse=True)
    total = sum(len(c) for c in clusters_sorted)
    t_target, v_target = total * train_r, total * val_r
    t_imgs, v_imgs, te_imgs = [], [], []
    t_count = v_count = 0
    for cluster in clusters_sorted:
        if t_count < t_target:
            t_imgs.extend(cluster); t_count += len(cluster)
        elif v_count < v_target:
            v_imgs.extend(cluster); v_count += len(cluster)
        else:
            te_imgs.extend(cluster)
    return t_imgs, v_imgs, te_imgs

print("\nBuilding cluster-based split:")
split_data = {"train": [], "val": [], "test": []}

for folder_name, class_name in new_class_map.items():
    class_folder = NEW_DATA_ROOT / folder_name
    t0 = time.time()
    clusters = cluster_class_folder(class_folder)
    if not clusters: continue
    t_imgs, v_imgs, te_imgs = cluster_split(clusters)
    method = "signer-folders" if any(d.is_dir() for d in class_folder.iterdir()) else "dhash"
    print(f"  {class_name:<12} {len(clusters):>4} clusters  train={len(t_imgs):<5} val={len(v_imgs):<4} test={len(te_imgs):<4}  ({method})  {time.time()-t0:.1f}s")
    for split, imgs in [("train",t_imgs),("val",v_imgs),("test",te_imgs)]:
        split_data[split].extend({"path": str(p), "class": class_name} for p in imgs)

for split, records in split_data.items():
    with open(SPLIT_DIR / f"{split}.json", "w") as f:
        json.dump(records, f)
    print(f"  Saved {split}.json ({len(records)} samples)")

train_n, val_n, test_n = len(split_data["train"]), len(split_data["val"]), len(split_data["test"])
total_n = train_n + val_n + test_n
print(f"\nTotal: {total_n}  train {train_n/total_n:.1%}  val {val_n/total_n:.1%}  test {test_n/total_n:.1%}")
assert train_n > 0 and val_n > 0 and test_n > 0


# Transforms
# Training on raw RGB (no CLAHE) to match real camera input at inference time.
# HorizontalFlip is excluded because mirroring changes the sign meaning.

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10, fill=0),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

class SignDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path) as f:
            self.samples = json.load(f)
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        item  = self.samples[i]
        img   = Image.open(item["path"]).convert("RGB")
        label = CLASS_TO_IDX[item["class"]]
        if self.transform: img = self.transform(img)
        return img, label


# Model: MobileNetV2 with pretrained ImageNet weights, classifier head replaced

def build_model(num_classes=N_CLASSES):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(1280, num_classes)
    return m

model = build_model().to(DEVICE)

total_p = sum(p.numel() for p in model.parameters())
print(f"\nParameters: {total_p:,} total")

with torch.no_grad():
    dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
    out   = model(dummy)
    assert out.shape == (2, N_CLASSES)
print(f"Forward pass OK: {dummy.shape} -> {out.shape}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# Training

BATCH_SIZE   = 64
NUM_WORKERS  = 4
HEAD_EPOCHS  = 5
FULL_EPOCHS  = 25
HEAD_LR      = 1e-3
FULL_LR      = 5e-5
WEIGHT_DECAY = 1e-4
PATIENCE     = 6
PIN_MEMORY   = DEVICE.type == "cuda"

train_ds = SignDataset(SPLIT_DIR / "train.json", transform=train_transform)
val_ds   = SignDataset(SPLIT_DIR / "val.json",   transform=eval_transform)
test_ds  = SignDataset(SPLIT_DIR / "test.json",  transform=eval_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

print(f"\nTrain: {len(train_ds)} ({len(train_loader)} batches)")
print(f"Val  : {len(val_ds)} ({len(val_loader)} batches)")
print(f"Test : {len(test_ds)} ({len(test_loader)} batches)")

def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss_sum = correct = total = 0
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)
            if is_train: optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(scaler is not None)):
                out  = model(imgs)
                loss = criterion(out, lbls)
            if is_train:
                (scaler.scale(loss) if scaler else loss).backward()
                if scaler: scaler.step(optimizer); scaler.update()
                else:      optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct  += (out.argmax(1) == lbls).sum().item()
            total    += imgs.size(0)
    return loss_sum / total, correct / total

def freeze_backbone(model):
    for p in model.features.parameters():    p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters(): p.requires_grad = True

history = {"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[],"lr":[]}

# Phase 1: train the classifier head only
print(f"\nPhase 1 - head only ({HEAD_EPOCHS} epochs, lr={HEAD_LR})")
freeze_backbone(model)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
scaler    = GradScaler(enabled=(DEVICE.type=="cuda"))
best_val_acc = 0.0

for ep in range(HEAD_EPOCHS):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler)
    va_loss, va_acc = run_epoch(model, val_loader,   criterion)
    scheduler.step(va_acc)
    lr_now = optimizer.param_groups[0]["lr"]
    history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
    history["lr"].append(lr_now)
    flag = ""
    if va_acc > best_val_acc:
        best_val_acc = va_acc; torch.save(model.state_dict(), MODEL_OUT); flag = " <- best"
    print(f"  ep {ep+1}/{HEAD_EPOCHS}  train {tr_loss:.4f}/{tr_acc*100:.2f}%  val {va_loss:.4f}/{va_acc*100:.2f}%  lr={lr_now:.2e}  {time.time()-t0:.1f}s{flag}")

# Phase 2: fine-tune all layers with lower lr
print(f"\nPhase 2 - full fine-tune ({FULL_EPOCHS} epochs, lr={FULL_LR})")
unfreeze_all(model)
optimizer    = optim.AdamW(model.parameters(), lr=FULL_LR, weight_decay=WEIGHT_DECAY)
scheduler    = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
patience_cnt = 0

for ep in range(FULL_EPOCHS):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler)
    va_loss, va_acc = run_epoch(model, val_loader,   criterion)
    scheduler.step(va_acc)
    lr_now = optimizer.param_groups[0]["lr"]
    history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
    history["lr"].append(lr_now)
    flag = ""
    if va_acc > best_val_acc:
        best_val_acc = va_acc; patience_cnt = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names":      TRAIN_CLASSES,
            "num_classes":      N_CLASSES,
            "best_val_acc":     best_val_acc,
            "epoch":            HEAD_EPOCHS + ep + 1,
            "mean":             MEAN,
            "std":              STD,
            "preprocessing":    "raw_rgb_no_clahe",
        }, MODEL_OUT)
        flag = " <- best"
    else:
        patience_cnt += 1
    print(f"  ep {ep+1}/{FULL_EPOCHS}  train {tr_loss:.4f}/{tr_acc*100:.2f}%  val {va_loss:.4f}/{va_acc*100:.2f}%  lr={lr_now:.2e}  patience={patience_cnt}/{PATIENCE}  {time.time()-t0:.1f}s{flag}")
    if patience_cnt >= PATIENCE:
        print("  Early stopping."); break

with open(HISTORY_OUT, "w") as f:
    json.dump(history, f, indent=2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
epochs = range(1, len(history["train_loss"]) + 1)
axes[0].plot(epochs, history["train_loss"], label="train"); axes[0].plot(epochs, history["val_loss"], label="val")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].axvline(HEAD_EPOCHS, color="gray", linestyle="--")
axes[1].plot(epochs, [a*100 for a in history["train_acc"]], label="train")
axes[1].plot(epochs, [a*100 for a in history["val_acc"]], label="val")
axes[1].set_title("Accuracy (%)"); axes[1].legend()
axes[2].plot(epochs, history["lr"]); axes[2].set_title("Learning Rate"); axes[2].set_yscale("log")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150); plt.close()

print(f"\nBest val accuracy: {best_val_acc*100:.2f}%")
print(f"Model saved to: {MODEL_OUT}")


# Evaluation

ck = torch.load(MODEL_OUT, map_location=DEVICE)
model.load_state_dict(ck["model_state_dict"])
model.eval()

def run_evaluation(model, loader, device):
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            out = model(imgs.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(lbls.numpy())
    return np.array(preds), np.array(labels)

def report_and_save(preds, labels, class_names, tag):
    acc    = (preds == labels).mean()
    report = classification_report(labels, preds, target_names=class_names, digits=3, zero_division=0)
    cm     = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    row    = cm.sum(axis=1, keepdims=True)
    cmn    = cm.astype(float) / np.where(row==0, 1, row)

    rpath = RESULTS_DIR / f"{tag}_report.txt"
    rpath.write_text(f"Accuracy: {acc*100:.4f}%\n\n{report}", encoding="utf-8")

    _, axes = plt.subplots(1, 2, figsize=(22, 10))
    for ax, data, title in zip(axes, [cm, cmn], ["Raw", "Normalized"]):
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_title(f"{tag}  Acc={acc*100:.2f}%  -  {title}", fontsize=10)
        t = np.arange(len(class_names))
        ax.set_xticks(t); ax.set_yticks(t)
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{tag}_confusion_matrix.png", dpi=150); plt.close()

    print(f"\n{tag.upper()}  -  Accuracy: {acc*100:.4f}%  ({(preds==labels).sum()}/{len(labels)})")
    print(report)
    return acc

# Sanity check on training data
print("\nSanity check (train subset):")
by_class = defaultdict(list)
for i, item in enumerate(train_ds.samples):
    by_class[item["class"]].append(i)
sanity_idx = []
for idxs in by_class.values():
    random.shuffle(idxs); sanity_idx.extend(idxs[:20])
sanity_loader = DataLoader(Subset(train_ds, sanity_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
s_preds, s_labels = run_evaluation(model, sanity_loader, DEVICE)
sanity_acc = (s_preds == s_labels).mean()
print(f"  Sanity accuracy: {sanity_acc*100:.2f}% (expected >= 90%)")
assert sanity_acc >= 0.90, f"Sanity check failed ({sanity_acc*100:.2f}%)"

# In-distribution test (held-out test split from same dataset)
print("\nIn-distribution test:")
te_preds, te_labels = run_evaluation(model, test_loader, DEVICE)
in_dist_acc = report_and_save(te_preds, te_labels, TRAIN_CLASSES, "in_distribution")

# Error grid
wrong_idx = [i for i,(p,l) in enumerate(zip(te_preds, te_labels)) if p != l]
random.shuffle(wrong_idx); wrong_idx = wrong_idx[:40]
if wrong_idx:
    cols = 8; rows = (len(wrong_idx)+cols-1)//cols
    _, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2.5))
    axes = np.array(axes).flatten()
    for ax in axes: ax.axis("off")
    _mean = np.array(MEAN); _std = np.array(STD)
    for ax, wi in zip(axes, wrong_idx):
        img, _ = test_ds[wi]
        npimg  = img.permute(1,2,0).numpy() * _std + _mean
        ax.imshow(npimg.clip(0,1)); ax.axis("off")
        ax.set_title(f"T:{TRAIN_CLASSES[te_labels[wi]]}\nP:{TRAIN_CLASSES[te_preds[wi]]}", fontsize=6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "in_distribution_errors.png", dpi=120); plt.close()

# OOD test using ArASL2018 (completely different dataset)
print("\nOOD test - ArASL2018:")

class OODDataset(Dataset):
    def __init__(self, root, class_to_idx, transform):
        self.transform = transform
        self.samples   = []
        for folder in sorted(root.iterdir()):
            if not folder.is_dir(): continue
            cls = resolve_class(folder.name)
            if cls not in class_to_idx: continue
            idx = class_to_idx[cls]
            for p in folder.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append((str(p), idx))
        print(f"  OOD samples: {len(self.samples)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

ood_ds     = OODDataset(ARASL_ROOT, CLASS_TO_IDX, eval_transform)
ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
ood_preds, ood_labels = run_evaluation(model, ood_loader, DEVICE)
ood_acc = report_and_save(ood_preds, ood_labels, TRAIN_CLASSES, "ood_arasl2018")

# Save outputs
with open(WORK_DIR / "class_names.json", "w") as f:
    json.dump(TRAIN_CLASSES, f, indent=2)

print("\nSaved files:")
for p in sorted(WORK_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(WORK_DIR)}  ({p.stat().st_size/1024:.1f} KB)")

print(f"\nFinal results:")
print(f"  In-distribution accuracy : {in_dist_acc*100:.2f}%")
print(f"  OOD accuracy (ArASL2018) : {ood_acc*100:.2f}%")
print(f"  Model: {MODEL_OUT}")
