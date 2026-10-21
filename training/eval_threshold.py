"""Precision / coverage of named predictions by confidence threshold for the self-trained model (v3 recipe).

Re-runs the 6 validation folds of autolabel_ft.py (same seeds, same splits) with the accepted pseudo-labels from
its CSV on the training side, but this time KEEPS the out-of-fold probabilities, then reports, for each threshold,
the precision of named predictions on held-out HUMAN labels. Pseudo-labelled crops of a validation clip are dropped.

Usage:  python eval_threshold.py --data export5 --csv autolabel_persona_ft.csv --out oof_v3.npz
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from autolabel import MIN_CLASS
from autolabel_ft import CATCH_ALL
from finetune import group_split, time_split
from train_final import fit, probabilities
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="oof_v3.npz")
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    data = Path(a.data)
    rows = [r for r in csv.DictReader(open(data.parent / f"{data.name}_manifest.csv")) if r["kind"] == "persona"]
    cnt = Counter(r["label"] for r in rows if r["label"])
    classes = sorted(c for c, n in cnt.items() if n >= MIN_CLASS)
    n_cls = len(classes)
    named = [classes.index(c) for c in classes if c not in CATCH_ALL]
    H = [i for i, r in enumerate(rows) if r["label"] in classes]
    by_id = {r["crop_id"]: i for i, r in enumerate(rows)}
    pseudo = {by_id[r["crop_id"]]: classes.index(r["auto_label"]) for r in csv.DictReader(open(a.csv))
              if r["auto_accepted"] == "1" and r["crop_id"] in by_id}
    imgs = [Image.open(data / r["path"]).convert("RGB") for r in rows]
    y_h = np.array([classes.index(rows[i]["label"]) for i in H])
    clip = [r["clip_id"] for r in rows]
    hclips = [clip[i] for i in H]
    htimes = [rows[i]["created_at"] for i in H]
    print(f"{len(H)} human, {len(pseudo)} pseudo-labelled crops", flush=True)

    oof = {"group": [], "time": []}
    for proto, s in [("group", 0), ("group", 1), ("group", 2), ("time", 0), ("time", 1), ("time", 2)]:
        t0 = time.time()
        tr, va = group_split(y_h, hclips, s) if proto == "group" else time_split(hclips, htimes, s)
        val_clips = {hclips[k] for k in va}
        ps = [(i, l) for i, l in pseudo.items() if clip[i] not in val_clips]
        net = fit([imgs[H[k]] for k in tr] + [imgs[i] for i, _ in ps],
                  np.array([y_h[k] for k in tr] + [l for _, l in ps]), n_cls, a.epochs, s)
        p = probabilities(net, [imgs[H[k]] for k in va])
        del net
        torch.cuda.empty_cache()
        oof[proto].append((y_h[va], p))
        print(f"  {proto} fold {s} done ({time.time() - t0:.0f}s)", flush=True)
    np.savez(a.out, **{f"{k}_{i}_{n}": v for k, parts in oof.items() for i, (yy, pp) in enumerate(parts)
                       for n, v in (("y", yy), ("p", pp))})

    print("\n=== v3 recipe: precision of named predictions by threshold (out-of-fold, human labels) ===")
    print(f"{'tau':>5s} | {'time: precision':>16s} {'coverage':>9s} | {'group: precision':>17s} {'coverage':>9s}")
    for tau in np.arange(0.20, 0.81, 0.05):
        cells = []
        for proto in ("time", "group"):
            yt = np.concatenate([yy for yy, _ in oof[proto]])
            pp = np.concatenate([q for _, q in oof[proto]])
            pred, conf = pp.argmax(1), pp.max(1)
            sel = np.isin(pred, named) & (conf >= tau)
            cells.append(((pred[sel] == yt[sel]).mean() if sel.sum() else float("nan"), sel.sum() / max(1, np.isin(yt, named).sum())))
        print(f"{tau:5.2f} | {cells[0][0] * 100:15.1f}% {cells[0][1] * 100:8.1f}% | {cells[1][0] * 100:16.1f}% {cells[1][1] * 100:8.1f}%")
    Path("eval_threshold.done").touch()


if __name__ == "__main__":
    main()
