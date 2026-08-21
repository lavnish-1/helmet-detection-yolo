"""
train_helmet_classifier.py

Fine-tunes MobileNetV2 (ImageNet-pretrained) on a small labeled dataset of
head crops to produce models/helmet_classifier.pt, which stage2 loads for
inference.

Expected dataset layout (standard torchvision ImageFolder format):

    data/
      with_helmet/     <- images of heads/riders wearing a helmet
      without_helmet/  <- images of heads/riders NOT wearing a helmet

Where to get labeled images (pick one):
  - Kaggle "Helmet Detection" dataset (andrewmvd/helmet-detection) -- has
    bounding-box annotations you can crop with the boxes to build the two
    folders above.
  - Crop head regions yourself from a few minutes of traffic footage using
    get_head_crop() in utils.py, and manually sort the crops into the two
    folders.
  - A public PPE/helmet Roboflow dataset exported in classification format.

Aim for at least ~150-200 images per class to start; more data and some
manual review of misclassified images is the main lever for improving
accuracy after the first training run.

Run:
    python train_helmet_classifier.py --epochs 15 --data_dir ../data
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from stage2_helmet_classifier import build_model, IMG_SIZE, CLASS_NAMES


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data",
                         help="Folder containing with_helmet/ and without_helmet/ subfolders")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="../models/helmet_classifier.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_tf, val_tf = get_transforms()
    full_dataset = datasets.ImageFolder(args.data_dir, transform=train_tf)

    # Sanity check: ImageFolder assigns class indices alphabetically, which
    # must match CLASS_NAMES in stage2_helmet_classifier.py or predictions
    # will be silently swapped.
    print("Detected classes (index order):", full_dataset.classes)
    assert full_dataset.classes == sorted(CLASS_NAMES), (
        f"Folder names {full_dataset.classes} don't match expected {sorted(CLASS_NAMES)}. "
        f"Rename your data/ subfolders to exactly 'with_helmet' and 'without_helmet'."
    )

    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_tf  # validation should not use augmentation

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

    print(f"Training complete. Best val_acc={best_val_acc:.3f}. Weights at {args.out}")


if __name__ == "__main__":
    main()
