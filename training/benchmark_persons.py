"""Benchmark for telling people apart, including the honest test of auto-labelling.

Methods (every number is measured on HUMAN labels only, never on pseudo-labels):
  LR  CLIP-L frozen + logistic regression, human labels only
  LR+ the same, plus pseudo-labels from the unlabelled pool
  FT  DINOv2-B fine-tuned, human labels only
  FT  CLIP-L fine-tuned, human labels only
  FT+ CLIP-L fine-tuned, human labels plus pseudo-labels

The pseudo-labels are regenerated INSIDE every fold from that fold's training labels only. The ones stored in
the database come from a model that saw all human labels, so reusing them here would leak the validation set.

Protocols: random held-out clips (group) and train-on-older / validate-on-newer (time).
Usage:  python benchmark_persons.py --data export3 --features feat_clip_L_921.npy [--group 3] [--time 3]
"""
import argparse
import csv
import json
import time
from collections import Counter

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import balanced_accuracy_score

from autolabel import MIN_CLASS, fit, predict, self_train
from finetune import group_split, pad_square, time_split

dev = "cuda"
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP = ((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))
ARCH = {
    "dinov2_B": ("vit_base_patch14_dinov2.lvd142m", {"img_size": 224}, IMAGENET, 32, False),
    "clip_L": ("vit_large_patch14_clip_224.openai", {}, CLIP, 16, True),
}
TAU = 0.75
NAMED = ("Domenico", "Io", "Lucrezia", "Mamma", "Papà", "Rossana")


class DS(torch.utils.data.Dataset):
    def __init__(self, imgs, ys, tf):
        self.imgs, self.ys, self.tf = imgs, ys, tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), int(self.ys[i])


def transforms(mean, std):
    ev = T.Compose([T.Lambda(pad_square), T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(mean, std)])
    tr = T.Compose([T.Lambda(pad_square), T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.85, 1.2)),
                    T.RandomHorizontalFlip(), T.RandomApply([T.ColorJitter(0.3, 0.3, 0.2)], p=0.8),
                    T.RandomGrayscale(0.3), T.ToTensor(), T.Normalize(mean, std)])
    return ev, tr


def finetune(model, imgs_tr, y_tr, imgs_va, n_cls, epochs, seed):
    arch, kw, (mean, std), bs, ckpt = ARCH[model]
    ev_tf, tr_tf = transforms(mean, std)
    torch.manual_seed(seed)
    bb = timm.create_model(arch, pretrained=True, num_classes=0, **kw)
    if ckpt:
        bb.set_grad_checkpointing(True)
    net = nn.Sequential(bb, nn.Linear(bb.num_features, n_cls)).to(dev)
    opt = torch.optim.AdamW([{"params": bb.parameters(), "lr": 1e-5 if ckpt else 2e-5},
                             {"params": net[1].parameters(), "lr": 1e-3}], weight_decay=0.05)
    steps = epochs * ((len(imgs_tr) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-5 if ckpt else 2e-5, 1e-3], total_steps=steps, pct_start=0.15)
    counts = np.bincount(y_tr, minlength=n_cls)
    weights = torch.tensor(len(y_tr) / (n_cls * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    dl = torch.utils.data.DataLoader(DS(imgs_tr, y_tr, tr_tf), batch_size=bs, shuffle=True, num_workers=4)
    for _ in range(epochs):
        net.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(net(x).float(), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(imgs_va), 32):
            x = torch.stack([ev_tf(im) for im in imgs_va[i:i + 32]]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds.append(net(x).argmax(1).cpu())
    del net, opt
    torch.cuda.empty_cache()
    return torch.cat(preds).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--features", required=True, help="cached CLIP-L features for the persona rows of the manifest")
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--time", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--skip", default="", help="comma list of methods to skip, e.g. FT_clipL+")
    a = ap.parse_args()
    from pathlib import Path
    data = Path(a.data)
    rows = [r for r in csv.DictReader(open(data.parent / f"{data.name}_manifest.csv")) if r["kind"] == "persona"]
    X = np.load(a.features)
    assert len(X) == len(rows), "features do not match the manifest"
    labels = [r["label"] for r in rows]
    cnt = Counter(l for l in labels if l)
    classes = sorted(c for c, n in cnt.items() if n >= MIN_CLASS)
    H = [i for i, l in enumerate(labels) if l in classes]
    U = [i for i, l in enumerate(labels) if l == ""]
    imgs = [Image.open(data / r["path"]).convert("RGB") for r in rows]
    yh = np.array([classes.index(labels[i]) for i in H])
    clips = [rows[i]["clip_id"] for i in H]
    times = [rows[i]["created_at"] for i in H]
    pool_clips = [rows[i]["clip_id"] for i in U]
    n_cls = len(classes)
    ok = {classes.index(c) for c in NAMED if c in classes}
    print(f"{len(H)} human-labelled crops, {len(U)} unlabelled; classes: " + ", ".join(f"{c}={cnt[c]}" for c in classes))
    skip = set(a.skip.split(","))
    methods = [m for m in ("LR_clipL", "LR_clipL+auto", "FT_dinoB", "FT_clipL", "FT_clipL+auto") if m not in skip]
    out = {p: {m: {"y": [], "p": [], "bacc": [], "acc": []} for m in methods} for p in ("group", "time")}

    folds = [("group", s) for s in range(a.group)] + [("time", s) for s in range(a.time)]
    for proto, s in folds:
        t0 = time.time()
        tr, va = group_split(yh, clips, s) if proto == "group" else time_split(clips, times, s)
        val_clips = {clips[i] for i in va}
        pool = [k for k, c in enumerate(pool_clips) if c not in val_clips]  # no near-duplicates of validation crops
        Xtr, ytr, Xva, yva = X[H][tr], yh[tr], X[H][va], yh[va]
        pred = {}
        pred["LR_clipL"] = predict(fit(Xtr, ytr, 10), Xva)[0]
        # pseudo-labels from THIS fold's training labels only
        Xpool = X[U][pool]
        _, acc, lab, _, _, hist = self_train(Xtr, ytr, Xpool, TAU, ok, 10)
        n_acc = int(acc.sum())
        pi = [k for k, ok_ in zip(pool, acc) if ok_]
        py = lab[acc]
        pred["LR_clipL+auto"] = predict(fit(np.vstack([Xtr, Xpool[acc]]), np.concatenate([ytr, py]), 10,
                                            np.concatenate([np.ones(len(ytr)), np.full(n_acc, 0.5)])), Xva)[0]
        tr_imgs = [imgs[H[i]] for i in tr]
        va_imgs = [imgs[H[i]] for i in va]
        if "FT_dinoB" in methods:
            pred["FT_dinoB"] = finetune("dinov2_B", tr_imgs, ytr, va_imgs, n_cls, a.epochs, s)
        if "FT_clipL" in methods:
            pred["FT_clipL"] = finetune("clip_L", tr_imgs, ytr, va_imgs, n_cls, a.epochs, s)
        if "FT_clipL+auto" in methods:
            pseudo_imgs = [imgs[U[k]] for k in pi]
            pred["FT_clipL+auto"] = finetune("clip_L", tr_imgs + pseudo_imgs, np.concatenate([ytr, py]), va_imgs,
                                             n_cls, a.epochs, s)
        for m in methods:
            r = out[proto][m]
            r["y"] += list(yva)
            r["p"] += list(pred[m])
            r["bacc"].append(balanced_accuracy_score(yva, pred[m]))
            r["acc"].append(float((pred[m] == yva).mean()))
        print(f"  {proto} fold {s}: train {len(tr)} val {len(va)} pseudo-labels accepted {n_acc}/{len(pool)} "
              f"({time.time() - t0:.0f}s)  bacc " + " | ".join(f"{m} {out[proto][m]['bacc'][-1] * 100:.0f}" for m in methods), flush=True)

    for proto in ("group", "time"):
        if not out[proto][methods[0]]["acc"]:
            continue
        print(f"\n=== {proto}: " + ("random held-out clips" if proto == "group" else "train on older clips, validate on newer") + " ===")
        print(f"{'method':18s} {'accuracy':>16s} {'balanced acc.':>16s} {'named-person recall':>20s}")
        for m in methods:
            r = out[proto][m]
            y, p = np.array(r["y"]), np.array(r["p"])
            named = np.isin(y, list(ok))
            print(f"{m:18s} {np.mean(r['acc']) * 100:6.1f}% ± {np.std(r['acc']) * 100:4.1f}  "
                  f"{np.mean(r['bacc']) * 100:6.1f}% ± {np.std(r['bacc']) * 100:4.1f}  {(y[named] == p[named]).mean() * 100:17.1f}%")
    best = max(methods, key=lambda m: np.mean(out["time"][m]["bacc"]) if out["time"][m]["bacc"] else 0)
    y, p = np.array(out["time"][best]["y"]), np.array(out["time"][best]["p"])
    print(f"\nbest on the time protocol: {best}\nconfusion (rows = true, columns = predicted):")
    print(" " * 20 + "".join(f"{c[:7]:>8s}" for c in classes))
    for i, c in enumerate(classes):
        row = [(p[y == i] == j).sum() for j in range(n_cls)]
        print(f"{c[:19]:19s}" + "".join(f"{v:8d}" for v in row) + (f"   recall {row[i] / max(1, sum(row)) * 100:5.1f}%" if sum(row) else ""))
    json.dump({proto: {m: {k: v for k, v in r.items() if k in ("bacc", "acc")} for m, r in ms.items()} for proto, ms in out.items()}
              | {"classes": classes}, open("results_persona_benchmark.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
