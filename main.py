# main.py
import os
import csv
import random
import argparse
import logging
from datetime import datetime
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from PIL import Image
import numpy as np

# ---------------------------
# Logging & Reproducibility
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("skin-train")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (slower):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# Dataset
# ---------------------------
class SkinDataset(Dataset):
    """
    Expects:
      - images_dir: directory with image files (e.g., .jpg/.png)
      - labels_csv: CSV with two columns: filename,label
      - class_map: dict mapping class_name -> class_index (e.g., {"melanoma":0, "nevus":1, ...})
    """
    def __init__(self, images_dir: str, labels_csv: str, class_map: Dict[str, int], transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.class_map = class_map
        self.samples = self._load_csv(labels_csv)

    def _load_csv(self, labels_csv: str) -> List[Tuple[str, int]]:
        samples = []
        with open(labels_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            # Expect header like: filename,label
            if len(header) < 2 or header[0].lower() != "filename" or header[1].lower() != "label":
                raise ValueError("labels_csv must have header: filename,label")
            for row in reader:
                fname, label_name = row[0].strip(), row[1].strip()
                if label_name not in self.class_map:
                    raise ValueError(f"Label '{label_name}' not in class_map.")
                label_idx = self.class_map[label_name]
                samples.append((fname, label_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, fname)
        # Use PIL for robust decoding
        with Image.open(img_path).convert("RGB") as img:
            if self.transform is not None:
                img = self.transform(img)
        return img, label

# ---------------------------
# Utilities
# ---------------------------
def split_indices(n: int, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    val_size = int(n * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices

def compute_metrics(outputs, targets) -> Dict[str, float]:
    # Outputs: logits [B, C]
    # Targets: [B]
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    acc = correct / total
    return {"accuracy": acc}

def save_checkpoint(state: Dict, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, path)
    logger.info(f"Saved checkpoint to: {path}")

# ---------------------------
# Training & Validation Loops
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_examples = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = labels.size(0)
        metrics = compute_metrics(outputs, labels)
        running_loss += loss.item() * batch_size
        running_acc += metrics["accuracy"] * batch_size
        n_examples += batch_size

    return {
        "loss": running_loss / n_examples,
        "accuracy": running_acc / n_examples
    }

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n_examples = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        metrics = compute_metrics(outputs, labels)
        running_loss += loss.item() * batch_size
        running_acc += metrics["accuracy"] * batch_size
        n_examples += batch_size

    return {
        "loss": running_loss / n_examples,
        "accuracy": running_acc / n_examples
    }

# ---------------------------
# Main
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a skin disease classifier (EfficientNet-B0).")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory with training images (e.g., ISIC dermoscopic images).")
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="CSV with columns filename,label.")
    parser.add_argument("--class_names", type=str, nargs="+", required=True,
                        help="List of class names in the order you want to index them, e.g., melanoma nevus bcc.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze EfficientNet backbone and train only classifier head.")
    parser.add_argument("--resize", type=int, default=224,
                        help="Image resize dimension (square).")
    return parser.parse_args()

def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    # Train-time augmentation helps generalization for skin lesion images
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=EfficientNet_B0_Weights.IMAGENET1K_V1.transforms().mean,
                             std=EfficientNet_B0_Weights.IMAGENET1K_V1.transforms().std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=EfficientNet_B0_Weights.IMAGENET1K_V1.transforms().mean,
                             std=EfficientNet_B0_Weights.IMAGENET1K_V1.transforms().std),
    ])
    return train_tf, val_tf

def build_model(n_classes: int, freeze_backbone: bool = False):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, n_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    return model

def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Class mapping
    class_map = {name: idx for idx, name in enumerate(args.class_names)}
    logger.info(f"Class map: {class_map}")

    # Transforms
    train_tf, val_tf = build_transforms(args.resize)

    # Full dataset
    full_ds = SkinDataset(images_dir=args.images_dir, labels_csv=args.labels_csv, class_map=class_map, transform=None)
    train_indices, val_indices = split_indices(len(full_ds), val_ratio=args.val_ratio, seed=args.seed)

    # Wrap with transforms per split
    # We create lightweight wrappers by reusing CSV rows via index filtering
    def subset_dataset(dataset: SkinDataset, indices: List[int], transform):
        ds = SkinDataset(dataset.images_dir, args.labels_csv, dataset.class_map, transform=transform)
        ds.samples = [dataset.samples[i] for i in indices]
        return ds

    train_ds = subset_dataset(full_ds, train_indices, train_tf)
    val_ds = subset_dataset(full_ds, val_indices, val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model, loss, optimizer
    model = build_model(n_classes=len(class_map), freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()

    # If freezing backbone, optimize only classifier; else all params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=args.grad_clip)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch:02d}/{args.epochs} | "
                    f"Train loss: {train_metrics['loss']:.4f}, acc: {train_metrics['accuracy']:.4f} | "
                    f"Val loss: {val_metrics['loss']:.4f}, acc: {val_metrics['accuracy']:.4f}")

        # Save last checkpoint
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "class_map": class_map,
            "val_acc": val_metrics["accuracy"],
            "args": vars(args)
        }, args.output_dir, f"last_{start_time}.pt")

        # Save best checkpoint
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "class_map": class_map,
                "val_acc": val_metrics["accuracy"],
                "args": vars(args)
            }, args.output_dir, f"best_{start_time}.pt")

    logger.info(f"Training complete. Best val acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
