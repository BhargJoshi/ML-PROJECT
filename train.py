"""
Bone Fracture Detection - PyTorch Version
Transfer Learning with ResNet50 + Pre-split Dataset
Optimized for RTX 5050 (8GB VRAM) with CUDA support
"""

# ── Truncated image fix ────────────────────────────────────────────────────
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
# ──────────────────────────────────────────────────────────────────────────

import os
import json
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.cuda.amp import GradScaler, autocast  # mixed precision

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 16      # fills 8GB VRAM nicely with float16
NUM_WORKERS   = 4       # parallel CPU data loading
EPOCHS        = 50
PHASE1_EPOCHS = 10      # frozen base, train head only
DATASET_DIR   = "Dataset"
MODEL_SAVE    = "model/fracture_model.pth"
RESUME_EPOCH  = 10    # set to last completed epoch number to resume, e.g. 11

os.makedirs("model",  exist_ok=True)
os.makedirs("static", exist_ok=True)

# ─────────────────────────────────────────────
# DEVICE: Force RTX, hide Intel iGPU
# ─────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        # On dual-GPU laptops, cuda:0 is always the discrete GPU (RTX 5050)
        device = torch.device("cuda:0")
        print(f"✅ Using GPU : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   CUDA      : {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("⚠️  No CUDA GPU found — running on CPU (will be slow)")
    return device

# ─────────────────────────────────────────────
# UTILITY: Scan and remove corrupt images
# ─────────────────────────────────────────────
def clean_dataset(dataset_dir):
    print("🔍 Scanning dataset for corrupt images...")
    removed = 0
    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            fpath = os.path.join(root, fname)
            try:
                with Image.open(fpath) as img:
                    img.verify()
            except Exception as e:
                print(f"  ❌ Removing: {fpath}  ({e})")
                os.remove(fpath)
                removed += 1
    print(f"✅ Scan complete. Removed {removed} corrupt file(s).\n")

# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
def build_loaders():
    # ImageNet mean/std — required for pretrained ResNet50
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), train_transforms)
    val_dataset   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"),   val_transforms)
    test_dataset  = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"),  val_transforms)

    # pin_memory=True speeds up CPU->GPU transfer on CUDA machines
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"📦 Train: {len(train_dataset)} images | "
          f"Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"   Classes: {train_dataset.class_to_idx}")

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(device):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Phase 1: freeze entire base
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head with our binary head
    in_features = model.fc.in_features
    # No Sigmoid here — BCEWithLogitsLoss applies it internally
    # in a numerically stable way that's safe with float16/autocast
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
    )
    return model.to(device)

def unfreeze_last_n_layers(model, n=20):
    """Unfreeze the last n parameter tensors of the ResNet base for fine-tuning."""
    base_params = []
    for name, param in model.named_parameters():
        if not name.startswith("fc."):   # exclude our custom head
            base_params.append(param)

    # Freeze all base params first
    for param in base_params:
        param.requires_grad = False

    # Unfreeze last n
    for param in base_params[-n:]:
        param.requires_grad = True

    # fc head always trains
    for param in model.fc.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params after unfreeze: {trainable:,}")

# ─────────────────────────────────────────────
# TRAINING LOOP (one epoch)
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, scaler, is_training):
    model.train() if is_training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            # Mixed precision forward pass — equivalent to TF mixed_float16
            with autocast():
                outputs = model(images).squeeze(1)
                loss    = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Apply sigmoid to logits for prediction (not done by model anymore)
            probs = torch.sigmoid(outputs.detach().cpu())
            preds = (probs > 0.5).float()
            all_preds.extend(probs.numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item() * images.size(0)
            correct    += (preds == labels.cpu()).sum().item()
            total      += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    auc      = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    return avg_loss, accuracy, auc

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train():
    # clean_dataset(DATASET_DIR)  # uncomment on first run to remove corrupt files

    device = get_device()
    train_loader, val_loader, test_loader, class_to_idx = build_loaders()

    with open("model/class_indices.json", "w") as f:
        json.dump(class_to_idx, f)

    # BCEWithLogitsLoss = Sigmoid + BCELoss fused together, float16-safe
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler()   # handles float16 gradient scaling

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  [],
               "train_auc":  [], "val_auc":  []}

    best_val_auc   = 0.0
    best_weights   = None
    patience_count = 0
    PATIENCE       = 5
    start_epoch    = 0

    # ── RESUME LOGIC ─────────────────────────────────────────────────────────
    if os.path.exists(MODEL_SAVE) and RESUME_EPOCH is not None:
        print(f"🔄 Resuming from '{MODEL_SAVE}' at epoch {RESUME_EPOCH}...")
        checkpoint = torch.load(MODEL_SAVE, map_location=device)
        model = build_model(device)
        unfreeze_last_n_layers(model, n=20)
        model.load_state_dict(checkpoint["model_state"])

        start_epoch  = checkpoint.get("epoch", RESUME_EPOCH)
        best_val_auc = checkpoint.get("best_val_auc", 0.0)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
        )
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        print(f"✅ Resumed. Best val AUC so far: {best_val_auc:.4f}")

    else:
        # ── FRESH TRAINING ────────────────────────────────────────────────────
        print("🧠 Building ResNet50 model from scratch...")
        model = build_model(device)

        # ── PHASE 1: Train head only ──────────────────────────────────────────
        print(f"\n🏋️  Phase 1: Training head only ({PHASE1_EPOCHS} epochs)...")
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

        for epoch in range(PHASE1_EPOCHS):
            tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, criterion,
                                                 optimizer, device, scaler, True)
            vl_loss, vl_acc, vl_auc = run_epoch(model, val_loader, criterion,
                                                 optimizer, device, scaler, False)
            scheduler.step(vl_auc)

            history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
            history["train_acc"].append(tr_acc);   history["val_acc"].append(vl_acc)
            history["train_auc"].append(tr_auc);   history["val_auc"].append(vl_auc)

            print(f"  Epoch {epoch+1:02d}/{PHASE1_EPOCHS} | "
                  f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
                  f"Acc {tr_acc:.4f}/{vl_acc:.4f} | "
                  f"AUC {tr_auc:.4f}/{vl_auc:.4f}")

            if vl_auc > best_val_auc:
                best_val_auc = vl_auc
                best_weights = copy.deepcopy(model.state_dict())

        # ── PHASE 2: Fine-tune last 20 layers ────────────────────────────────
        print("\n🔧 Phase 2: Fine-tuning last 20 ResNet50 layers...")
        unfreeze_last_n_layers(model, n=20)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
        )
        start_epoch = PHASE1_EPOCHS

    # ── PHASE 2 LOOP (also runs when resuming) ────────────────────────────────
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    for epoch in range(start_epoch, EPOCHS):
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, criterion,
                                             optimizer, device, scaler, True)
        vl_loss, vl_acc, vl_auc = run_epoch(model, val_loader, criterion,
                                             optimizer, device, scaler, False)
        scheduler.step(vl_auc)

        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(vl_acc)
        history["train_auc"].append(tr_auc);   history["val_auc"].append(vl_auc)

        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc {tr_acc:.4f}/{vl_acc:.4f} | "
              f"AUC {tr_auc:.4f}/{vl_auc:.4f}")

        # Save best checkpoint with optimizer state (enables clean resume)
        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch":           epoch + 1,
                "model_state":     best_weights,
                "optimizer_state": optimizer.state_dict(),
                "best_val_auc":    best_val_auc,
            }, MODEL_SAVE)
            print(f"  💾 Saved best model (val AUC: {best_val_auc:.4f})")
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs.")
            break

    if best_weights:
        model.load_state_dict(best_weights)

    plot_history(history)
    print("\n🏁 Evaluating on hidden TEST dataset...")
    evaluate_model(model, test_loader, device)

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device, non_blocking=True)
            outputs = torch.sigmoid(model(images).squeeze(1)).cpu().numpy()
            all_preds.extend(outputs)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    y_pred     = (all_preds > 0.5).astype(int)

    with open("model/class_indices.json") as f:
        class_to_idx = json.load(f)
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]

    print("\n📊 Final Test Report:")
    print(classification_report(all_labels, y_pred, target_names=class_names))
    print(f"🎯 ROC-AUC Score: {roc_auc_score(all_labels, all_preds):.4f}")

    cm = confusion_matrix(all_labels, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_title("Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig("static/confusion_matrix.png")
    print("✅ Results saved to static/ folder.")

# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
def plot_history(history):
    epochs_range = range(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs_range, history["train_acc"], label="Train Accuracy")
    axes[0].plot(epochs_range, history["val_acc"],   label="Val Accuracy")
    axes[0].set_title("Accuracy"); axes[0].legend()

    axes[1].plot(epochs_range, history["train_auc"], label="Train AUC")
    axes[1].plot(epochs_range, history["val_auc"],   label="Val AUC")
    axes[1].set_title("AUC"); axes[1].legend()

    axes[2].plot(epochs_range, history["train_loss"], label="Train Loss")
    axes[2].plot(epochs_range, history["val_loss"],   label="Val Loss")
    axes[2].set_title("Loss"); axes[2].legend()

    for ax in axes:
        ax.axvline(x=PHASE1_EPOCHS, color="gray", linestyle="--",
                   alpha=0.6, label="Fine-tune start")
        ax.set_xlabel("Epoch")

    plt.suptitle("Training Progress — Phase 1 | Phase 2", fontsize=13)
    plt.tight_layout()
    plt.savefig("static/training_curves.png")
    print("📈 Training curves saved to static/training_curves.png")

if __name__ == "__main__":
    train()