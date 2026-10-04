"""Benchmark: tell individual cats / people apart from crops.

  A) frozen DINOv2 features  ->  kNN rule (what the web UI uses today)  and  logistic regression
  B) full fine-tuning of DINOv2 with augmentation

Validation is split BY CLIP (crops of one clip are near-duplicates), repeated over several seeds.
Usage:  python finetune.py --data export --kind animale [--splits 5] [--epochs 15]
"""
import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

MODELS = {"S": "vit_small_patch14_dinov2.lvd142m", "B": "vit_base_patch14_dinov2.lvd142m"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
dev = "cuda"


def pad_square(img, fill=(124, 116, 104)):
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), fill)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


EVAL_TF = T.Compose([T.Lambda(pad_square), T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
TRAIN_TF = T.Compose([
    T.Lambda(pad_square),
    T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.85, 1.2)),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.3, 0.3, 0.2)], p=0.8),
    T.RandomGrayscale(0.3),  # night images are infrared grayscale
    T.ToTensor(), T.Normalize(MEAN, STD),
])


class DS(torch.utils.data.Dataset):
    def __init__(self, imgs, ys, tf):
        self.imgs, self.ys, self.tf = imgs, ys, tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), self.ys[i]


def load(data_dir, kind):
    rows = [r for r in csv.DictReader(open(data_dir.parent / f"{data_dir.name}_manifest.csv"))
            if r["kind"] == kind and r["identity"] == "1"]
    classes = sorted({r["label"] for r in rows})
    imgs = [Image.open(data_dir / r["path"]).convert("RGB") for r in rows]
    ys = [classes.index(r["label"]) for r in rows]
    clips = [r["clip_id"] for r in rows]
    return classes, imgs, np.array(ys), clips


def group_split(ys, clips, seed, val_frac=0.25):
    """~val_frac of each class's clips go to validation; a clip is never in both sets."""
    rng = random.Random(seed)
    per_class = defaultdict(set)
    for y, c in zip(ys, clips):
        per_class[y].add(c)
    val_clips = set()
    for y, cs in per_class.items():
        cs = sorted(cs); rng.shuffle(cs)
        val_clips.update(cs[:max(1, round(len(cs) * val_frac))])
    val = [i for i, c in enumerate(clips) if c in val_clips]
    train = [i for i, c in enumerate(clips) if c not in val_clips]
    return train, val


@torch.no_grad()
def features(model, imgs):
    model.eval()
    out = []
    for i in range(0, len(imgs), 64):
        x = torch.stack([EVAL_TF(im) for im in imgs[i:i + 64]]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(x).float().cpu())
    f = torch.cat(out).numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def knn_predict(ftr, ytr, fva, k=3):
    """Same rule as the web suggestion: per class mean of the top-k similarities."""
    classes = sorted(set(ytr))
    sims = fva @ ftr.T
    scores = np.stack([np.sort(sims[:, ytr == c], axis=1)[:, ::-1][:, :k].mean(axis=1) for c in classes], axis=1)
    return np.array(classes)[scores.argmax(1)]


def finetune(name, imgs, ys, tr, va, n_cls, epochs, seed, lr_backbone=2e-5, lr_head=1e-3):
    torch.manual_seed(seed)
    backbone = timm.create_model(MODELS[name], pretrained=True, num_classes=0, img_size=224)
    net = nn.Sequential(backbone, nn.Linear(backbone.num_features, n_cls)).to(dev)
    opt = torch.optim.AdamW([{"params": backbone.parameters(), "lr": lr_backbone},
                             {"params": net[1].parameters(), "lr": lr_head}], weight_decay=0.05)
    steps = epochs * ((len(tr) + 31) // 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_backbone, lr_head], total_steps=steps, pct_start=0.15)
    counts = np.bincount(ys[tr], minlength=n_cls)
    weights = torch.tensor(len(tr) / (n_cls * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    dl = torch.utils.data.DataLoader(DS([imgs[i] for i in tr], ys[tr], TRAIN_TF), batch_size=32,
                                     shuffle=True, num_workers=4, drop_last=False)
    for ep in range(epochs):
        net.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(net(x).float(), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(va), 64):
            x = torch.stack([EVAL_TF(imgs[j]) for j in va[i:i + 64]]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds.append(net(x).argmax(1).cpu())
    return torch.cat(preds).numpy(), loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--kind", default="animale")
    ap.add_argument("--splits", type=int, default=5); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--models", default="S,B")
    a = ap.parse_args()
    classes, imgs, ys, clips = load(Path(a.data), a.kind)
    n = len(classes)
    print(f"{a.kind}: {len(imgs)} crops, {len(set(clips))} clips, classes: " +
          ", ".join(f"{c}={int((ys == i).sum())}" for i, c in enumerate(classes)))
    results = defaultdict(lambda: {"acc": [], "bacc": [], "y": [], "p": []})
    for name in a.models.split(","):
        backbone = timm.create_model(MODELS[name], pretrained=True, num_classes=0, img_size=224).to(dev)
        feats = features(backbone, imgs)
        del backbone
        for s in range(a.splits):
            tr, va = group_split(ys, clips, s)
            ytr, yva = ys[tr], ys[va]
            runs = {
                f"A  {name} frozen + kNN(top3)": knn_predict(feats[tr], ytr, feats[va]),
                f"A  {name} frozen + logistic regr.": LogisticRegression(C=10, max_iter=3000, class_weight="balanced")
                    .fit(feats[tr], ytr).predict(feats[va]),
            }
            t = time.time()
            runs[f"B  {name} fine-tuned ({a.epochs} ep)"], _ = finetune(name, imgs, ys, tr, va, n, a.epochs, s)
            for key, p in runs.items():
                r = results[key]
                r["acc"].append((p == yva).mean()); r["bacc"].append(balanced_accuracy_score(yva, p))
                r["y"] += list(yva); r["p"] += list(p)
            print(f"  model {name} split {s}: train={len(tr)} val={len(va)}  (finetune {time.time() - t:.0f}s)", flush=True)
    print(f"\n=== {a.kind}: {a.splits} splits per clip, validation ≈ 25% ===")
    print(f"{'method':38s} {'accuracy':>16s} {'balanced acc.':>16s}")
    for key, r in results.items():
        print(f"{key:38s} {np.mean(r['acc']) * 100:6.1f}% ± {np.std(r['acc']) * 100:4.1f}  {np.mean(r['bacc']) * 100:6.1f}% ± {np.std(r['bacc']) * 100:4.1f}")
    best = max(results, key=lambda k: np.mean(results[k]["bacc"]))
    y, p = np.array(results[best]["y"]), np.array(results[best]["p"])
    print(f"\nBest: {best}\nRecall per class / confusion (rows = true, columns = predicted):")
    print(" " * 12 + "".join(f"{c[:8]:>9s}" for c in classes))
    for i, c in enumerate(classes):
        row = [(p[y == i] == j).sum() for j in range(n)]
        print(f"{c[:11]:11s}" + "".join(f"{v:9d}" for v in row) + f"   recall {row[i] / max(1, sum(row)) * 100:5.1f}%")
    json.dump({k: {"acc": v["acc"], "bacc": v["bacc"]} for k, v in results.items()} | {"classes": classes},
              open(f"results_{a.kind}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
