"""
run_experiments.py  -  Baselines, ablations and variance for the helmet classifier
(camera-ready revision, Paper 2112, AISUMMIT-2026)

  Configurations (each answers a reviewer request):
    main          MobileNetV2, ImageNet-pretrained, augmentation, 128x128 (5 seeds -> variance)
    resnet18      ResNet-18 backbone            -> baseline: heavier CNN
    effnet_b0     EfficientNet-B0 backbone      -> baseline: modern light CNN
    no_pretrain   MobileNetV2 from scratch      -> ablation: value of transfer learning
    no_aug        MobileNetV2, no augmentation  -> ablation: value of augmentation
    res224        MobileNetV2 at 224x224        -> ablation: input resolution

  python run_experiments.py --quick     # 1-minute smoke test
  python run_experiments.py             # full run (20 runs)
  python run_experiments.py --grouped   # stricter photo-wise split, main model x5
"""
import argparse, csv, os, random, shutil, statistics, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ["with_helmet", "without_helmet"]          # label 0, label 1
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CONFIGS = {
    #  name          backbone       pretrained  augment  size   seeds
    "main":        ("mobilenet_v2", True,       True,    128,   [0, 1, 2, 3, 4]),
    "resnet18":    ("resnet18",     True,       True,    128,   [0, 1, 2]),
    "effnet_b0":   ("efficientnet_b0", True,    True,    128,   [0, 1, 2]),
    "no_pretrain": ("mobilenet_v2", False,      True,    128,   [0, 1, 2]),
    "no_aug":      ("mobilenet_v2", True,       False,   128,   [0, 1, 2]),
    "res224":      ("mobilenet_v2", True,       True,    224,   [0, 1, 2]),
}


def load_split(split_dir):
    """Reads the fixed split already on disk (data_split/{train,val,test}/{class})."""
    split = {}
    for part in ("train", "val", "test"):
        items = []
        for label, cls in enumerate(CLASSES):
            folder = Path(split_dir) / part / cls
            if not folder.is_dir():
                raise SystemExit(f"Missing folder: {folder}. Run split_dataset.py first.")
            items += [(str(p), label) for p in sorted(folder.iterdir())
                      if p.suffix.lower() in IMG_EXT]
        split[part] = items
    return split


def grouped_split(split_dir, seed=42, frac=(0.70, 0.15)):
    """Re-splits all crops so every crop from the same source photo lands in the same split."""
    import re
    pool = load_split(split_dir)
    items = pool["train"] + pool["val"] + pool["test"]
    groups = {}
    for p, y in items:
        stem = re.sub(r"_\d+$", "", Path(p).stem)       # 'photo_3.jpg' -> 'photo'
        groups.setdefault(stem, []).append((p, y))
    keys = sorted(groups); random.Random(seed).shuffle(keys)
    total, split, seen = len(items), {"train": [], "val": [], "test": []}, 0
    for k in keys:
        part = "train" if seen < frac[0] * total else ("val" if seen < (frac[0] + frac[1]) * total else "test")
        split[part] += groups[k]; seen += len(groups[k])
    return split


class CropDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def transforms_for(size, augment):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])
    if not augment:
        return test_tf, test_tf
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), norm])
    return train_tf, test_tf


def build_model(backbone, pretrained):
    if backbone == "mobilenet_v2":
        m = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif backbone == "resnet18":
        m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        raise ValueError(backbone)
    return m


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); ys, ps, confs = [], [], []
    for x, y in loader:
        prob = torch.softmax(model(x.to(device)), dim=1).cpu()
        c, p = prob.max(dim=1)
        ys += y.tolist(); ps += p.tolist(); confs += c.tolist()
    return ys, ps, confs


def train_one(cfg_name, seed, split, args, device):
    backbone, pretrained, augment, size, _ = CONFIGS[cfg_name]
    set_seed(seed)
    train_tf, test_tf = transforms_for(size, augment)
    g = torch.Generator(); g.manual_seed(seed)
    tr = DataLoader(CropDataset(split["train"], train_tf), batch_size=16, shuffle=True,
                    num_workers=0, generator=g)
    va = DataLoader(CropDataset(split["val"], test_tf), batch_size=64, num_workers=0)
    te = DataLoader(CropDataset(split["test"], test_tf), batch_size=64, num_workers=0)

    model = build_model(backbone, pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for x, y in tr:
            opt.zero_grad(); loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward(); opt.step()
        yv, pv, _ = predict(model, va, device)
        acc = accuracy_score(yv, pv)
        if acc >= best_acc:                     # same rule as train_final.py; VALIDATION only
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    yt, pt, ct = predict(model, te, device)     # test set touched once, after training
    p, r, f, _ = precision_recall_fscore_support(yt, pt, average="macro", zero_division=0)
    res = dict(config=cfg_name, seed=seed, val_acc=best_acc,
               test_acc=accuracy_score(yt, pt), macro_p=p, macro_r=r, macro_f1=f,
               cm=confusion_matrix(yt, pt, labels=[0, 1]).tolist(),
               params_M=sum(t.numel() for t in model.parameters()) / 1e6,
               train_min=(time.time() - t0) / 60)
    return res, (yt, pt, ct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default="../data_split")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--quick", action="store_true", help="1 seed, 2 epochs, smoke test")
    ap.add_argument("--grouped", action="store_true",
                    help="group-wise split (no source photo shared between splits); runs 'main' only")
    ap.add_argument("--out", default="../outputs/experiments")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else
          "  <- no GPU found; this will be slow. Install the CUDA build of PyTorch."))

    if args.grouped:
        split = grouped_split(args.split_dir)
        args.configs = "main"
        args.out = args.out.rstrip("/\\") + "_grouped"
        out = Path(args.out); out.mkdir(exist_ok=True)
        print("GROUP-WISE split: no source photo appears in more than one split.")
    else:
        split = load_split(args.split_dir)
    for k in split:
        n0 = sum(1 for _, y in split[k] if y == 0)
        print(f"  {k:5s}: {len(split[k]):4d} images  ({n0} with_helmet, {len(split[k]) - n0} without_helmet)")

    if args.quick:
        args.epochs = 2
    runs = []
    for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        seeds = CONFIGS[name][4][:1] if args.quick else CONFIGS[name][4]
        for seed in seeds:
            print(f"\n=== {name}  seed {seed} ===", flush=True)
            res, preds = train_one(name, seed, split, args, device)
            print(f"  val_acc={res['val_acc']:.4f}  test_acc={res['test_acc']:.4f}  "
                  f"macro_f1={res['macro_f1']:.4f}  ({res['train_min']:.1f} min)  cm={res['cm']}")
            runs.append(res)
            if name == "main" and seed == 0:
                save_predictions(split["test"], preds, out)
            write_outputs(runs, out)           # write after every run, so nothing is lost
    print(f"\nDone. See {out.resolve()}")


def save_predictions(test_items, preds, out):
    yt, pt, ct = preds
    mis = out / "misclassified"; shutil.rmtree(mis, ignore_errors=True); mis.mkdir()
    with open(out / "main_seed0_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "true", "pred", "confidence", "correct"])
        for (p, _), y, pr, c in zip(test_items, yt, pt, ct):
            w.writerow([p, CLASSES[y], CLASSES[pr], f"{c:.3f}", y == pr])
            if y != pr:
                shutil.copy(p, mis / f"true-{CLASSES[y]}__pred-{CLASSES[pr]}__conf{c:.2f}__{Path(p).name}")


def write_outputs(runs, out):
    keys = ["test_acc", "macro_p", "macro_r", "macro_f1", "val_acc"]
    with open(out / "all_runs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "seed"] + keys + ["params_M", "train_min", "confusion_matrix"])
        for r in runs:
            w.writerow([r["config"], r["seed"]] + [f"{r[k]:.4f}" for k in keys] +
                       [f"{r['params_M']:.2f}", f"{r['train_min']:.2f}", r["cm"]])
    rows, latex = [], []
    for name in dict.fromkeys(r["config"] for r in runs):
        rs = [r for r in runs if r["config"] == name]
        def ms(k):
            v = [r[k] * 100 for r in rs]
            return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
        stats = {k: ms(k) for k in keys}
        rows.append([name, len(rs)] + [f"{m:.2f} +/- {s:.2f}" for m, s in stats.values()] +
                    [f"{rs[0]['params_M']:.2f}"])
        latex.append(f"{name} & {rs[0]['params_M']:.1f} & "
                     f"{stats['test_acc'][0]:.1f} $\\pm$ {stats['test_acc'][1]:.1f} & "
                     f"{stats['macro_f1'][0]:.1f} $\\pm$ {stats['macro_f1'][1]:.1f} \\\\")
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "runs"] + [k + " (%)" for k in keys] + ["params_M"])
        w.writerows(rows)
    open(out / "latex_table.tex", "w").write("\n".join(latex) + "\n")


if __name__ == "__main__":
    main()
