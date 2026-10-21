"""Frozen-feature benchmark of candidate backbones for telling people apart (human labels only, many folds).

Each backbone is run at its native resolution with horizontal-flip averaging; a logistic regression is scored
on 10 clip-grouped and 3 temporal folds. Pairs of the best backbones are concatenated too.
Features are cached next to the data as feat_bb_<name>_<n>.npy.

Usage:  python backbone_bench.py --data export5
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from autolabel import MIN_CLASS
from finetune import group_split, pad_square, time_split

dev = "cuda"
CANDIDATES = {
    "clip_L": "vit_large_patch14_clip_224.openai",
    "dinov2_L": "vit_large_patch14_dinov2.lvd142m",
    "siglip2_L": "vit_large_patch16_siglip_gap_256.v2_webli",
    "siglip2_so400m": "vit_so400m_patch14_siglip_gap_224.v2_webli",
    "pe_core_L": "vit_pe_core_large_patch14_336.fb",
    "dinov3_L": "vit_large_patch16_dinov3.lvd1689m",
    "eva02_clip_L": "eva02_large_patch14_clip_224.merged2b",
}


@torch.no_grad()
def extract(arch, imgs, half=True):
    kw = {"img_size": 224} if "dinov2" in arch else {}
    m = timm.create_model(arch, pretrained=True, num_classes=0, **kw).eval().to(dev)
    m = m.half() if half else m
    cfg = timm.data.resolve_data_config({}, model=m)
    size = (224, 224) if "dinov2" in arch else cfg["input_size"][1:]
    tf = T.Compose([T.Lambda(pad_square), T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    out = []
    for i in range(0, len(imgs), 16):
        x = torch.stack([tf(im) for im in imgs[i:i + 16]]).to(dev)
        x = x.half() if half else x
        out.append(((m(x).float() + m(torch.flip(x, dims=[3])).float()) / 2).cpu())
    f = torch.cat(out).numpy()
    del m
    torch.cuda.empty_cache()
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def score(X, y, clips, times, C=10, n_group=10, n_time=3):
    res = {"group": [], "time": []}
    for kind, n in (("group", n_group), ("time", n_time)):
        for s in range(n):
            tr, va = group_split(y, clips, s) if kind == "group" else time_split(clips, times, s)
            if not va or len(set(y[tr])) < 2:
                continue
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
            res[kind].append(balanced_accuracy_score(y[va], clf.predict(X[va])))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    data = Path(a.data)
    rows = [r for r in csv.DictReader(open(data.parent / f"{data.name}_manifest.csv")) if r["kind"] == "persona"]
    cnt = Counter(r["label"] for r in rows if r["label"])
    classes = sorted(c for c, n in cnt.items() if n >= MIN_CLASS)
    H = [i for i, r in enumerate(rows) if r["label"] in classes]
    imgs = [Image.open(data / rows[i]["path"]).convert("RGB") for i in H]
    y = np.array([classes.index(rows[i]["label"]) for i in H])
    clips, times = [rows[i]["clip_id"] for i in H], [rows[i]["created_at"] for i in H]
    feats, table = {}, {}
    for name, arch in CANDIDATES.items():
        cache = data.parent / f"feat_bb_{name}_{len(H)}.npy"
        try:
            feats[name] = np.load(cache) if cache.exists() else extract(arch, imgs)
            if not np.isfinite(feats[name]).all():  # fp16 overflow in some backbones
                feats[name] = extract(arch, imgs, half=False)
            np.save(cache, feats[name])
        except Exception as e:  # gated weights, missing model...
            print(f"{name:16s} SKIPPED ({type(e).__name__}: {str(e)[:90]})", flush=True)
            continue
        table[name] = score(feats[name], y, clips, times)
        print(f"{name:16s} dim {feats[name].shape[1]:5d}  group {table[name]['group'][0] * 100:5.1f} ± {table[name]['group'][1] * 100:4.1f}"
              f"  time {table[name]['time'][0] * 100:5.1f} ± {table[name]['time'][1] * 100:4.1f}", flush=True)
    best = sorted(table, key=lambda k: -(table[k]["group"][0] + table[k]["time"][0]))[:4]
    for i in range(len(best)):
        for j in range(i + 1, len(best)):
            name = best[i] + "+" + best[j]
            table[name] = score(np.hstack([feats[best[i]], feats[best[j]]]), y, clips, times)
            print(f"{name:34s} group {table[name]['group'][0] * 100:5.1f} ± {table[name]['group'][1] * 100:4.1f}"
                  f"  time {table[name]['time'][0] * 100:5.1f} ± {table[name]['time'][1] * 100:4.1f}", flush=True)
    json.dump(table, open("results_backbone_bench.json", "w"), indent=1)
    Path("backbone_bench.done").touch()


if __name__ == "__main__":
    main()
