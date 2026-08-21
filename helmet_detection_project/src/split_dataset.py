"""
split_dataset.py

Creates a fixed, reproducible train/val/test split from data/with_helmet
and data/without_helmet, and physically copies files into:
    data_split/train/with_helmet/, data_split/train/without_helmet/
    data_split/val/with_helmet/,   data_split/val/without_helmet/
    data_split/test/with_helmet/,  data_split/test/without_helmet/

Why this exists (and why it's separate from train_helmet_classifier.py's
old built-in split): the original training script called
torch.utils.data.random_split() with no fixed seed, so every training run
validated on a different random subset. That's fine for quick iteration,
but it means the reported accuracy isn't reproducible and there's no
data that was NEVER touched during training/model-selection -- which is
exactly what a paper's Results section needs to report a defensible,
one-time-measured number.

This script uses a fixed random seed (42) and writes a manifest file
(data_split/split_manifest.json) recording exactly which original file
went into which split, so the split is fully reproducible and citable
in the paper's Methodology section (e.g. "70/15/15 train/val/test split,
stratified by class, fixed seed=42").

Usage:
    python split_dataset.py --data_dir ../data --out_dir ../data_split
"""

import argparse
import json
import os
import random
import shutil

CLASSES = ["with_helmet", "without_helmet"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--out_dir", default="../data_split")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    # test_ratio is whatever remains
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    manifest = {"seed": args.seed, "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio, "splits": {}}

    for cls in CLASSES:
        src_dir = os.path.join(args.data_dir, cls)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"Expected folder '{src_dir}' to exist.")

        files = sorted([f for f in os.listdir(src_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        random.shuffle(files)  # shuffled deterministically by the fixed seed

        n = len(files)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            out_folder = os.path.join(args.out_dir, split_name, cls)
            os.makedirs(out_folder, exist_ok=True)
            for f in split_files:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(out_folder, f))
            manifest["splits"].setdefault(split_name, {})[cls] = split_files

        print(f"{cls}: {n} total -> train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Split written to {args.out_dir}/{{train,val,test}}/")
    print(f"Manifest (exact file lists, for reproducibility) saved to {manifest_path}")
    print("\nIMPORTANT: retrain the classifier using ONLY data_split/train and "
          "data_split/val from now on (see train_final.py), and evaluate final "
          "metrics on data_split/test exactly once with evaluate.py.")


if __name__ == "__main__":
    main()
