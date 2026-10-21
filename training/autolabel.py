"""Auto-label the unlabelled crops of one kind, learning round by round (meant for the GPU server).

Human labels are the only ground truth: they choose the feature extractor, calibrate the confidence
threshold and evaluate everything. Pseudo-labels only ever join the TRAINING side of later rounds.

  1. extract features with several big frozen backbones (DINOv2 B/L/G, CLIP-L, and pairs of them)
  2. pick the best combination on human labels (clip-grouped splits + temporal splits)
  3. pick the confidence threshold by SIMULATING the whole procedure: hide the labels of some human crops,
     auto-label them, and count how many are right (true precision, not an estimate)
  4. self-train on the real unlabelled crops: each round fits on human + accepted pseudo-labels, re-predicts
  5. write autolabel_<kind>.csv: label, confidence, accepted yes/no; the rest is a review queue, least sure first

Usage:  python autolabel.py --data export3 --kind persona [--precision 0.95]
"""
import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from finetune import group_split, pad_square, time_split

dev = "cuda"
MIN_CLASS = 8  # human-labelled classes with fewer examples are ignored
BACKBONES = {
    "dinov2_B": ("vit_base_patch14_dinov2.lvd142m", {"img_size": 224}),
    "dinov2_L": ("vit_large_patch14_dinov2.lvd142m", {"img_size": 224}),
    "dinov2_G": ("vit_giant_patch14_dinov2.lvd142m", {"img_size": 224}),
    "clip_L": ("vit_large_patch14_clip_224.openai", {}),
}
COMBOS = ["dinov2_L+clip_L", "dinov2_G+clip_L", "dinov2_B+dinov2_L+clip_L"]


# ---------------------------------------------------------------- features
@torch.no_grad()
def extract(name, imgs):
    arch, kw = BACKBONES[name]
    m = timm.create_model(arch, pretrained=True, num_classes=0, **kw).eval().to(dev).half()
    cfg = timm.data.resolve_data_config({}, model=m)
    tf = T.Compose([T.Lambda(pad_square), T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    out = []
    for i in range(0, len(imgs), 32):
        x = torch.stack([tf(im) for im in imgs[i:i + 32]]).to(dev).half()
        out.append(((m(x).float() + m(torch.flip(x, dims=[3])).float()) / 2).cpu())  # horizontal-flip averaging
    f = torch.cat(out).numpy()
    del m
    torch.cuda.empty_cache()
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def get_features(name, imgs, cache):
    path = cache / f"feat_{name}_{len(imgs)}.npy"
    if path.exists():
        return np.load(path)
    f = extract(name, imgs)
    np.save(path, f)
    return f


# ---------------------------------------------------------------- model + evaluation
def fit(X, y, C, w=None):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, y, sample_weight=w)


def predict(clf, X):
    p = clf.predict_proba(X)
    best = p.argmax(1)
    return clf.classes_[best], p.max(1), p


def oof(Xh, yh, clips, times, C, n_group=5, n_time=3):
    """Out-of-fold predictions on human labels only: {'group': [(y_true, y_pred, conf)], 'time': [...]}."""
    res = {"group": [], "time": []}
    for kind, n in (("group", n_group), ("time", n_time)):
        for s in range(n):
            tr, va = group_split(yh, clips, s) if kind == "group" else time_split(clips, times, s)
            if not va or len(set(yh[tr])) < 2:
                continue
            pred, conf, _ = predict(fit(Xh[tr], yh[tr], C), Xh[va])
            res[kind].append((yh[va], pred, conf))
    return res


def summarize_oof(res):
    out = {}
    for kind, parts in res.items():
        out[kind + "_bacc"] = float(np.mean([balanced_accuracy_score(t, p) for t, p, _ in parts]))
        out[kind + "_acc"] = float(np.mean([(t == p).mean() for t, p, _ in parts]))
    out["score"] = (out["group_bacc"] + out["time_bacc"]) / 2
    return out


# ---------------------------------------------------------------- self-training
def self_train(Xh, yh, Xu, tau, ok_classes, C, rounds=6, pseudo_weight=0.5):
    """Fit on human labels, accept confident predictions on Xu, refit with them, repeat until stable."""
    clf = fit(Xh, yh, C)
    accepted = np.zeros(len(Xu), bool)
    labels = np.full(len(Xu), -1)
    history = []
    for r in range(rounds):
        lab, conf, _ = predict(clf, Xu)
        new = (conf >= tau) & np.isin(lab, list(ok_classes))
        changed = int((new != accepted).sum() + ((lab != labels) & new & accepted).sum())
        accepted, labels = new, lab
        history.append((r + 1, int(accepted.sum()), changed))
        if changed <= max(1, 0.01 * len(Xu)):
            break
        X = np.vstack([Xh, Xu[accepted]])
        y = np.concatenate([yh, labels[accepted]])
        w = np.concatenate([np.ones(len(yh)), np.full(int(accepted.sum()), pseudo_weight)])
        clf = fit(X, y, C, w)
    lab, conf, proba = predict(clf, Xu)
    return clf, accepted, lab, conf, proba, history


def simulate(Xh, yh, clips, Xu, tau, ok_classes, C, seeds=5, mask_frac=0.3):
    """True precision of the whole procedure: hide the labels of some human crops, auto-label, check."""
    right = shown = masked_total = 0
    base, selft = [], []
    for s in range(seeds):
        tr, va = group_split(yh, clips, s)
        tr_clips = sorted({clips[i] for i in tr})
        random.Random(100 + s).shuffle(tr_clips)
        hidden = set(tr_clips[:int(len(tr_clips) * mask_frac)])
        known = [i for i in tr if clips[i] not in hidden]
        mask = [i for i in tr if clips[i] in hidden]
        if not mask or len(set(yh[known])) < 2:
            continue
        pool = np.vstack([Xu, Xh[mask]]) if len(Xu) else Xh[mask]
        clf, acc, lab, _, _, _ = self_train(Xh[known], yh[known], pool, tau, ok_classes, C)
        sl = slice(len(Xu), None)
        right += int((lab[sl][acc[sl]] == yh[mask][acc[sl]]).sum())
        shown += int(acc[sl].sum())
        masked_total += len(mask)
        base.append(balanced_accuracy_score(yh[va], predict(fit(Xh[known], yh[known], C), Xh[va])[0]))
        selft.append(balanced_accuracy_score(yh[va], predict(clf, Xh[va])[0]))
    return {"precision": right / shown if shown else float("nan"), "coverage": shown / max(1, masked_total),
            "accepted": shown, "bacc_human_only": float(np.mean(base)), "bacc_self_trained": float(np.mean(selft))}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--kind", default="persona")
    ap.add_argument("--precision", type=float, default=0.95, help="target precision of auto-accepted labels")
    ap.add_argument("--backbones", default=",".join(BACKBONES))
    a = ap.parse_args()
    data = Path(a.data)
    rows = [r for r in csv.DictReader(open(data.parent / f"{data.name}_manifest.csv")) if r["kind"] == a.kind]
    labels = [r["label"] for r in rows]
    cnt = Counter(l for l in labels if l)
    classes = sorted(c for c, n in cnt.items() if n >= MIN_CLASS)
    H = [i for i, l in enumerate(labels) if l in classes]
    U = [i for i, l in enumerate(labels) if l == ""]
    print(f"{a.kind}: {len(H)} human-labelled crops in {len(classes)} classes, {len(U)} unlabelled")
    print("classes:", ", ".join(f"{c}={cnt[c]}" for c in classes))
    imgs = [Image.open(data / r["path"]).convert("RGB") for r in rows]
    yh = np.array([classes.index(labels[i]) for i in H])
    clips = [rows[i]["clip_id"] for i in H]
    times = [rows[i]["created_at"] for i in H]

    # 1. features
    feats = {}
    for name in a.backbones.split(","):
        t = time.time()
        try:
            feats[name] = get_features(name, imgs, data.parent)
            print(f"  features {name}: dim {feats[name].shape[1]}  ({time.time() - t:.0f}s)", flush=True)
        except Exception as e:  # missing weights, out of memory...
            print(f"  features {name}: SKIPPED ({type(e).__name__}: {str(e)[:100]})")
    for combo in COMBOS:
        parts = combo.split("+")
        if all(p in feats for p in parts):
            feats[combo] = np.hstack([feats[p] for p in parts])

    # 2. choose the best feature set on human labels
    print(f"\n=== feature sets (logistic regression C=10, human labels only) ===")
    print(f"{'features':34s} {'grouped bacc':>13s} {'temporal bacc':>14s} {'grouped acc':>12s} {'temporal acc':>13s}")
    table = {}
    for name, X in feats.items():
        s = summarize_oof(oof(X[H], yh, clips, times, 10))
        table[name] = s
        print(f"{name:34s} {s['group_bacc'] * 100:12.1f}% {s['time_bacc'] * 100:13.1f}% {s['group_acc'] * 100:11.1f}% {s['time_acc'] * 100:12.1f}%")
    best = max(table, key=lambda k: table[k]["score"])
    X = feats[best]
    Cs = {C: summarize_oof(oof(X[H], yh, clips, times, C))["score"] for C in (1, 10, 100)}
    C = max(Cs, key=Cs.get)
    print(f"\nbest: {best}  (C={C}; score by C: " + ", ".join(f"{k}: {v * 100:.1f}%" for k, v in Cs.items()) + ")")
    if not U:
        print("no unlabelled crops: nothing to auto-label")
        return

    # 3. threshold, calibrated on out-of-fold human predictions, then verified by simulating the procedure
    res = oof(X[H], yh, clips, times, C)
    pool = [(t, p, c) for parts in res.values() for (t, p, c) in parts]
    ytrue, ypred, conf = (np.concatenate([x[i] for x in pool]) for i in range(3))
    tau0 = 0.5
    for tau in np.arange(0.5, 1.0, 0.01):
        m = conf >= tau
        if m.sum() >= 30 and (ytrue[m] == ypred[m]).mean() >= a.precision:
            tau0 = float(tau)
            break
    else:
        tau0 = 0.9
    print(f"\nout-of-fold threshold for precision {a.precision:.0%}: {tau0:.2f}")
    print("precision by predicted class at that threshold:")
    ok_classes = set()
    for ci, c in enumerate(classes):
        m = (conf >= tau0) & (ypred == ci)
        prec = (ytrue[m] == ci).mean() if m.sum() else float("nan")
        good = m.sum() >= 8 and prec >= a.precision - 0.05
        if good:
            ok_classes.add(ci)
        print(f"   {c:22s} n={int(m.sum()):3d}  precision {prec * 100:5.1f}%  -> {'auto-accept' if good else 'REVIEW BY HAND'}")
    Xh, Xu = X[H], X[U]
    chosen = None
    print("\n=== simulation of the whole procedure (labels of some human crops hidden, then auto-labelled) ===")
    for tau in sorted({round(max(tau0, t), 2) for t in (tau0, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99)}):
        s = simulate(Xh, yh, clips, Xu, tau, ok_classes, C)
        print(f"  tau {tau:.2f}: precision {s['precision'] * 100:5.1f}%  coverage {s['coverage'] * 100:5.1f}%  accepted {s['accepted']:3d}"
              f" | val balanced acc: human only {s['bacc_human_only'] * 100:.1f}% -> self-trained {s['bacc_self_trained'] * 100:.1f}%", flush=True)
        if chosen is None and s["precision"] >= a.precision:
            chosen = (tau, s)
    if chosen is None:
        print("\n!! no threshold reaches the target precision in simulation: accepting nothing automatically.")
        tau, sim = 1.01, None
    else:
        tau, sim = chosen
    print(f"\nchosen threshold: {tau:.2f}")

    # 4. self-train on the real unlabelled crops
    clf, accepted, lab, conf_u, proba, hist = self_train(Xh, yh, Xu, tau, ok_classes, C)
    print("rounds (round, accepted, changed):", hist)
    order = np.argsort(proba, axis=1)
    second = clf.classes_[order[:, -2]]
    second_conf = np.take_along_axis(proba, order[:, -2:-1], axis=1)[:, 0]
    margin = conf_u - second_conf

    # 5. output
    out = []
    for k, i in enumerate(U):
        out.append({"crop_id": rows[i]["crop_id"], "path": rows[i]["path"], "auto_label": classes[lab[k]],
                    "auto_conf": round(float(conf_u[k]), 3), "accepted": int(accepted[k]),
                    "second_label": classes[second[k]], "second_conf": round(float(second_conf[k]), 3),
                    "margin": round(float(margin[k]), 3)})
    out.sort(key=lambda r: (r["accepted"], r["margin"]))  # review queue first: least sure at the top
    with open(f"autolabel_{a.kind}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    acc_counts = Counter(r["auto_label"] for r in out if r["accepted"])
    print(f"\nauto-accepted {int(accepted.sum())}/{len(U)} unlabelled crops:", dict(acc_counts))
    print(f"left for human review: {len(U) - int(accepted.sum())}")
    json.dump({"features": best, "C": C, "tau": tau, "target_precision": a.precision, "simulation": sim,
               "accepted": int(accepted.sum()), "total_unlabelled": len(U), "classes": classes,
               "feature_table": table}, open(f"autolabel_{a.kind}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
