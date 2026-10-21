"""Compare pseudo-label strategies on cached features, scored ONLY against human labels (clip-grouped splits).

Strategies (from the state-of-the-art review in docs/research_notes.md):
  lr          baseline: logistic regression, one global confidence threshold
  lr_pc       same, with a per-class threshold (FreeMatch-style class-adaptive: each class gets the threshold
              that reaches the target precision on out-of-fold predictions)
  spread      label spreading on a kNN graph over human + unlabelled crops (transductive, Iscen et al. CVPR'19)
  blend       average of lr and spread probabilities, per-class thresholds
  blend_knn   blend + open-set reject: crops far from every human-labelled crop are never accepted (Sun et al. ICML'22)

For each strategy the script prints precision and coverage of the accepted pseudo-labels on held-out human crops.

Usage:  python autolabel_v2.py --data export5 --kind persona --backbone vit_small_patch14_dinov2.lvd142m
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading

from finetune import group_split, pad_square

MIN_CLASS = 8
TARGETS = (0.90, 0.95)


@torch.no_grad()
def extract(arch, imgs, dev):
    m = timm.create_model(arch, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    cfg = timm.data.resolve_data_config({}, model=m)
    tf = T.Compose([T.Lambda(pad_square), T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    out = []
    for i in range(0, len(imgs), 32):
        x = torch.stack([tf(im) for im in imgs[i:i + 32]]).to(dev)
        out.append(((m(x) + m(torch.flip(x, dims=[3]))) / 2).cpu())  # horizontal-flip averaging
    f = torch.cat(out).numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def lr_proba(Xtr, ytr, Xte, n_cls, C=10):
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    p = np.zeros((len(Xte), n_cls))
    p[:, clf.classes_] = clf.predict_proba(Xte)
    return p


def spread_proba(Xl, yl, Xu, n_cls, k=10, alpha=0.8):
    X = np.vstack([Xl, Xu])
    y = np.concatenate([yl, np.full(len(Xu), -1)])
    ls = LabelSpreading(kernel="knn", n_neighbors=k, alpha=alpha, max_iter=100).fit(X, y)
    p = np.zeros((len(Xu), n_cls))
    p[:, ls.classes_] = ls.label_distributions_[len(Xl):]
    return p


def knn_dist(Xl, Xq, k=5):
    """Mean cosine distance of each query to its k nearest labelled crops (features are L2-normalised)."""
    s = np.sort(Xq @ Xl.T, axis=1)[:, -k:]
    return 1 - s.mean(1)


def class_thresholds(p, y, target, floor=0.5):
    """Per-class threshold: lowest confidence at which predictions of that class reach `target` precision."""
    pred, conf = p.argmax(1), p.max(1)
    th = np.full(p.shape[1], 1.01)
    for c in range(p.shape[1]):
        m = pred == c
        for t in np.arange(floor, 1.0, 0.01):
            s = m & (conf >= t)
            if s.sum() >= 5 and (y[s] == c).mean() >= target:
                th[c] = t
                break
    return th


def evaluate(Xh, yh, clips, Xu, n_cls, folds=5, seeds=2):
    """Hide the labels of held-out clips, auto-label them from the rest (+ unlabelled pool), count right/wrong."""
    names = ["lr", "lr_pc", "spread", "blend", "blend_knn"]
    acc = {(n, t): [0, 0] for n in names for t in TARGETS}  # [right, accepted]
    total = 0
    for s in range(folds * seeds):
        tr, va = group_split(yh, clips, s)
        if not va or len(set(yh[tr])) < 2:
            continue
        # thresholds come from an inner split of the training side, never from the held-out crops
        itr, iva = group_split(yh[tr], [clips[i] for i in tr], 100 + s)
        itr, iva = [tr[i] for i in itr], [tr[i] for i in iva]
        pool = np.vstack([Xu, Xh[va]])
        sl = slice(len(Xu), None)

        def probs(train, test_rows):
            Xte = np.vstack([Xu, Xh[test_rows]])
            p_lr = lr_proba(Xh[train], yh[train], Xte, n_cls)
            p_sp = spread_proba(Xh[train], yh[train], Xte, n_cls)
            return {"lr": p_lr, "spread": p_sp, "blend": (p_lr + p_sp) / 2}, Xte

        inner, Xin = probs(itr, iva)
        outer, _ = probs(tr, va)
        d_in = knn_dist(Xh[itr], Xh[iva])
        d_out = knn_dist(Xh[tr], Xh[va])
        reject = np.quantile(d_in, 0.95)  # keep 95% of the (known) validation crops
        total += len(va)
        y_in, y_out = yh[iva], yh[va]
        for t in TARGETS:
            for name in names:
                base = {"lr_pc": "lr", "blend_knn": "blend"}.get(name, name)
                p_in, p_out = inner[base][sl], outer[base][sl]
                if name in ("lr_pc", "blend", "spread", "blend_knn"):
                    th = class_thresholds(p_in, y_in, t)
                else:
                    th = np.full(n_cls, next((x for x in np.arange(0.5, 1.0, 0.01)
                                              if ((p_in.argmax(1) == y_in)[p_in.max(1) >= x]).mean() >= t
                                              and (p_in.max(1) >= x).sum() >= 5), 0.99))
                pred, conf = p_out.argmax(1), p_out.max(1)
                ok = conf >= th[pred]
                if name == "blend_knn":
                    ok &= d_out <= reject
                acc[(name, t)][0] += int((pred[ok] == y_out[ok]).sum())
                acc[(name, t)][1] += int(ok.sum())
    print(f"\nheld-out human crops evaluated: {total}")
    print(f"{'strategy':12s}" + "".join(f"  prec@{t:.2f}  cover" for t in TARGETS))
    for n in names:
        line = f"{n:12s}"
        for t in TARGETS:
            r, a = acc[(n, t)]
            line += f"  {r / a * 100 if a else float('nan'):8.1f}%  {a / total * 100:4.0f}%"
        print(line)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--kind", default="persona")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    a = ap.parse_args()
    data = Path(a.data)
    rows = [r for r in csv.DictReader(open(data.parent / f"{data.name}_manifest.csv")) if r["kind"] == a.kind]
    labels = [r["label"] for r in rows]
    cnt = Counter(l for l in labels if l)
    classes = sorted(c for c, n in cnt.items() if n >= MIN_CLASS)
    H = [i for i, l in enumerate(labels) if l in classes]
    U = [i for i, l in enumerate(labels) if l == ""]
    print(f"{a.kind}: {len(H)} human-labelled crops in {len(classes)} classes, {len(U)} unlabelled")
    cache = data.parent / f"feat_v2_{a.backbone}_{len(rows)}.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        imgs = [Image.open(data / r["path"]).convert("RGB") for r in rows]
        X = extract(a.backbone, imgs, dev)
        np.save(cache, X)
    yh = np.array([classes.index(labels[i]) for i in H])
    evaluate(X[H], yh, [rows[i]["clip_id"] for i in H], X[U], len(classes))


if __name__ == "__main__":
    main()
