"""
evaluate.py

Runs the trained classifier ONCE against the held-out test set
(data_split/test, produced by split_dataset.py and never touched during
training or model selection) and reports the metrics you'd actually cite
in a paper's Results section:

  - Accuracy, Precision, Recall, F1-score (per-class and macro-averaged)
  - Confusion matrix (printed as text, and saved as a PNG image)
  - Per-image predictions saved to a CSV (useful for picking example
    correct/incorrect figures for the paper)

Run this exactly once per model version you intend to report. If you
retrain the model afterward, that's a new model and you should treat any
new evaluate.py run as a fresh result, not a "best of" pick -- running it
multiple times and reporting only the best number would misrepresent the
model's real performance.

Usage:
    python evaluate.py --split_dir ../data_split --weights ../models/helmet_classifier.pt
"""

import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import numpy as np

from stage2_helmet_classifier import build_model, CLASS_NAMES
from train_helmet_classifier import get_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", default="../data_split")
    parser.add_argument("--weights", default="../models/helmet_classifier.pt")
    parser.add_argument("--out_dir", default="../outputs/evaluation")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, eval_tf = get_transforms()
    test_dir = os.path.join(args.split_dir, "test")
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
    assert test_ds.classes == sorted(CLASS_NAMES), (
        f"Test folder classes {test_ds.classes} don't match expected {sorted(CLASS_NAMES)}."
    )
    print(f"Test set: {len(test_ds)} images across classes {test_ds.classes}")

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels, all_confidences = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_confidences.extend(confs.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    class_names = test_ds.classes  # ['with_helmet', 'without_helmet'], sorted

    print("\n=== Test Set Results (held-out, evaluated once) ===")
    print(f"Overall accuracy: {acc:.4f}")
    for i, cname in enumerate(class_names):
        print(f"  {cname:16s} | precision={precision[i]:.3f}  recall={recall[i]:.3f}  "
              f"f1={f1[i]:.3f}  support={support[i]}")
    print(f"  {'macro avg':16s} | precision={macro_p:.3f}  recall={macro_r:.3f}  f1={macro_f1:.3f}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"{'':16s}" + "".join(f"{c:>16s}" for c in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:16s}" + "".join(f"{v:>16d}" for v in row))

    # Save confusion matrix as an image for the paper
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (Test Set, n={len(test_ds)})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)

    # Save full sklearn classification report as text
    report_path = os.path.join(args.out_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test set size: {len(test_ds)}\n")
        f.write(f"Overall accuracy: {acc:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
        f.write("\nConfusion matrix (rows=actual, cols=predicted):\n")
        f.write(f"{'':16s}" + "".join(f"{c:>16s}" for c in class_names) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{class_names[i]:16s}" + "".join(f"{v:>16d}" for v in row) + "\n")

    # Save per-image predictions (useful for picking figure examples)
    csv_path = os.path.join(args.out_dir, "predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "predicted_label", "confidence", "correct"])
        for (path, true_idx), pred_idx, conf in zip(test_ds.samples, all_preds, all_confidences):
            writer.writerow([path, class_names[true_idx], class_names[pred_idx],
                              f"{conf:.4f}", true_idx == pred_idx])

    print(f"\nSaved: {cm_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
