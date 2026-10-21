"""Paired, multi-fold fine-tune evaluation for CLIP-L and PE Core-L.

Evaluates human-labelled persona crops only. Group seeds are configurable;
temporal folds use the project's three predefined chronological windows.
Writes per-crop ensemble predictions and per-class recalls for error analysis.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score

from autolabel import MIN_CLASS
from finetune import group_split, time_split
from ft_backbones import train, probs

ARCHS = {
    "clip_L": ("vit_large_patch14_clip_224.openai", {}),
    "pe_core_L": ("vit_pe_core_large_patch14_336.fb", {}),
}


def rolling_time_split(clips, times, cut, window=0.2):
    """Train on a chronological prefix and validate on the following window."""
    first = {}
    for clip, timestamp in zip(clips, times):
        first[clip] = min(first.get(clip, timestamp), timestamp)
    order = sorted(first, key=lambda clip: first[clip])
    lo, hi = round(len(order) * cut), round(len(order) * (cut + window))
    train_clips, val_clips = set(order[:lo]), set(order[lo:hi])
    return ([i for i, clip in enumerate(clips) if clip in train_clips],
            [i for i, clip in enumerate(clips) if clip in val_clips])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--archs", default="clip_L,pe_core_L")
    ap.add_argument("--group-seeds", default="0,1,2,3,4")
    ap.add_argument("--time-folds", default="0,1,2")
    ap.add_argument("--extra-time-cuts", default="",
                    help="optional rolling-origin training fractions, e.g. 0.4,0.5; each validates on the next 20%%")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = Path(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(data.parent / f"{data.name}_manifest.csv"))
            if r["kind"] == "persona"]
    counts = Counter(r["label"] for r in rows if r["label"])
    classes = sorted(c for c, n in counts.items() if n >= MIN_CLASS)
    human = [r for r in rows if r["label"] in classes]
    imgs = [Image.open(data / r["path"]).convert("RGB") for r in human]
    y = np.asarray([classes.index(r["label"]) for r in human])
    clips = [r["clip_id"] for r in human]
    times = [r["created_at"] for r in human]
    folds = [("group", int(s), None) for s in a.group_seeds.split(",") if s.strip()]
    folds += [("time", int(s), None) for s in a.time_folds.split(",") if s.strip()]
    folds += [("time_extra", i, float(cut)) for i, cut in enumerate(a.extra_time_cuts.split(",")) if cut.strip()]
    summary = {"classes": classes, "n_human": len(human), "epochs": a.epochs,
               "seeds_per_fold": a.seeds, "folds": {}}
    pred_arrays = {}
    print(f"human crops={len(human)} classes={len(classes)} folds={len(folds)}", flush=True)
    for alias in a.archs.split(","):
        if alias not in ARCHS:
            raise ValueError(f"unknown architecture alias: {alias}")
        arch, kw = ARCHS[alias]
        summary["folds"][alias] = {}
        for protocol, fold, cut in folds:
            key = f"{alias}_{protocol}_{fold}"
            start = time.time()
            if protocol == "group":
                tr, va = group_split(y, clips, fold)
            elif protocol == "time":
                tr, va = time_split(clips, times, fold)
            else:
                tr, va = rolling_time_split(clips, times, cut)
            ps = []
            for seed in range(a.seeds):
                net, ev_tf = train(arch, kw, [imgs[i] for i in tr], y[tr], len(classes),
                                   a.epochs, 100 * fold + seed)
                ps.append(probs(net, [imgs[i] for i in va], ev_tf))
                del net
                torch.cuda.empty_cache()
            p = np.mean(ps, axis=0)
            pred = p.argmax(axis=1)
            recalls = recall_score(y[va], pred, labels=np.arange(len(classes)),
                                   average=None, zero_division=0)
            summary["folds"][alias][f"{protocol}_{fold}"] = {
                "n_train": len(tr), "n_val": len(va),
                "accuracy": float(accuracy_score(y[va], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y[va], pred)),
                "per_class_recall": {c: float(recalls[i]) for i, c in enumerate(classes)},
                "elapsed_seconds": round(time.time() - start, 1),
            }
            pred_arrays[f"{key}_row_index"] = np.asarray(va, dtype=np.int32)
            pred_arrays[f"{key}_y"] = y[va]
            pred_arrays[f"{key}_p"] = p.astype(np.float32)
            print(f"{alias} {protocol} fold {fold}: bacc={summary['folds'][alias][f'{protocol}_{fold}']['balanced_accuracy']*100:.1f}% "
                  f"acc={summary['folds'][alias][f'{protocol}_{fold}']['accuracy']*100:.1f}% ({time.time()-start:.0f}s)", flush=True)
    summary["human_rows"] = [{k: r.get(k, "") for k in ("crop_id", "clip_id", "camera", "created_at", "path", "label")}
                             for r in human]
    json.dump(summary, open(out / "metrics.json", "w"), indent=2, ensure_ascii=False)
    np.savez_compressed(out / "predictions.npz", **pred_arrays)
    Path(out / ".done").touch()
    print(f"saved {out}/metrics.json and predictions.npz", flush=True)


if __name__ == "__main__":
    main()
