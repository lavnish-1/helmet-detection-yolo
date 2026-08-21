"""
train_final.py

Trains the helmet classifier on the FIXED split produced by
split_dataset.py (data_split/train and data_split/val), rather than the
old random_split-with-no-seed approach in train_helmet_classifier.py.

This is the version whose result you'd actually report/cite in a paper:
- Trains only on data_split/train
- Picks the best checkpoint using data_split/val (never trained on)
- Never looks at data_split/test at all -- that stays untouched until
  evaluate.py runs on it exactly once, at the end.

Usage:
    python train_final.py --split_dir ../data_split --epochs 15
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from stage2_helmet_classifier import build_model, CLASS_NAMES
from train_helmet_classifier import get_transforms  # reuse the same augmentation setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", default="../data_split",
                         help="Folder containing train/ and val/ subfolders, from split_dataset.py")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="../models/helmet_classifier.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_tf, val_tf = get_transforms()

    train_dir = os.path.join(args.split_dir, "train")
    val_dir = os.path.join(args.split_dir, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    print("Train classes (index order):", train_ds.classes)
    print("Val classes (index order):  ", val_ds.classes)
    assert train_ds.classes == sorted(CLASS_NAMES) == val_ds.classes, (
        f"Class folder names must be exactly {sorted(CLASS_NAMES)} in both "
        f"train/ and val/, and must match between the two."
    )
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else 0.0

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved new best model (val_acc={val_acc:.3f}) to {args.out}")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.3f}. Weights at {args.out}")
    print("Next: run evaluate.py on data_split/test to get the final, reportable metrics.")


if __name__ == "__main__":
    main()
