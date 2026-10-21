"""Train a person model from HUMAN labels; CLIP-L remains the default.

1. 6 validation folds (3 random held-out clips + 3 train-on-older/validate-on-newer) give out-of-fold
   probabilities, used to choose the confidence threshold below which the watcher says "not recognised".
2. The final model is trained on every human label and saved as a checkpoint for the watcher.

Usage:  python train_final.py --data export5 [--model clip_L|pe_core_L] [--out candidate.pt]
"""
import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score

from autolabel import MIN_CLASS
from benchmark_persons import ARCH, DS, NAMED, transforms
from finetune import group_split, time_split

dev = "cuda"
MODEL = "clip_L"


def fit(imgs, y, n_cls, epochs, seed, model=MODEL):
    arch, kw, _, bs, ckpt = ARCH[model]
    torch.manual_seed(seed)
    bb = timm.create_model(arch, pretrained=True, num_classes=0, **kw)
    bb.set_grad_checkpointing(ckpt)
    cfg = timm.data.resolve_data_config({}, model=bb)
    mean, std = cfg["mean"], cfg["std"]
    size = cfg["input_size"][1:]
    _, tr_tf = transforms(mean, std, size)
    net = nn.Sequential(bb, nn.Linear(bb.num_features, n_cls)).to(dev)
    lr_b = 1e-5
    opt = torch.optim.AdamW([{"params": bb.parameters(), "lr": lr_b}, {"params": net[1].parameters(), "lr": 1e-3}], weight_decay=0.05)
    steps = epochs * ((len(imgs) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_b, 1e-3], total_steps=steps, pct_start=0.15)
    counts = np.bincount(y, minlength=n_cls)
    weights = torch.tensor(len(y) / (n_cls * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    dl = torch.utils.data.DataLoader(DS(imgs, y, tr_tf), batch_size=bs, shuffle=True, num_workers=4)
    for _ in range(epochs):
        net.train()
        for x, t in dl:
            x, t = x.to(dev), t.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(net(x).float(), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


@torch.no_grad()
def probabilities(net, imgs):
    cfg = timm.data.resolve_data_config({}, model=net[0])
    ev_tf, _ = transforms(cfg["mean"], cfg["std"], cfg["input_size"][1:])
    out = []
    for i in range(0, len(imgs), 32):
        x = torch.stack([ev_tf(im) for im in imgs[i:i + 32]]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.softmax(net(x).float(), dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    ap.add_argument("--model", choices=sorted(ARCH), default=MODEL)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--precision", type=float, default=0.90, help="target precision of named predictions")
    ap.add_argument("--artifact-smoke", action="store_true",
                    help="skip cross-validation; the checkpoint is only for loader/inference smoke tests")
    a = ap.parse_args()
    if a.out is None:
        a.out = "persona_clipL_v1.pt" if a.model == "clip_L" else f"persona_{a.model}_v1.pt"
    data = Path(a.data)
    rows = [r for r in csv.DictReader(open(data.parent / f"{data.name}_manifest.csv")) if r["kind"] == "persona"]
    cnt = Counter(r["label"] for r in rows if r["label"])
    classes = sorted(c for c, n in cnt.items() if n >= MIN_CLASS)
    H = [r for r in rows if r["label"] in classes]
    imgs = [Image.open(data / r["path"]).convert("RGB") for r in H]
    y = np.array([classes.index(r["label"]) for r in H])
    clips = [r["clip_id"] for r in H]
    times = [r["created_at"] for r in H]
    n_cls = len(classes)
    named = [classes.index(c) for c in NAMED if c in classes]
    print(f"{len(H)} human-labelled crops, {n_cls} classes: " + ", ".join(f"{c}={cnt[c]}" for c in classes), flush=True)

    chosen = None
    if a.artifact_smoke:
        print("Artifact smoke mode: skipping OOF evaluation and threshold selection; do not deploy this checkpoint.", flush=True)
    else:
        # 1. out-of-fold probabilities
        oof = {"group": [], "time": []}
        for proto, s in [("group", 0), ("group", 1), ("group", 2), ("time", 0), ("time", 1), ("time", 2)]:
            t0 = time.time()
            tr, va = group_split(y, clips, s) if proto == "group" else time_split(clips, times, s)
            net = fit([imgs[i] for i in tr], y[tr], n_cls, a.epochs, s, a.model)
            p = probabilities(net, [imgs[i] for i in va])
            del net
            torch.cuda.empty_cache()
            oof[proto].append((y[va], p))
            print(f"  {proto} fold {s}: train {len(tr)} val {len(va)}  bacc "
                  f"{balanced_accuracy_score(y[va], p.argmax(1)) * 100:.1f}%  ({time.time() - t0:.0f}s)", flush=True)
        np.savez("oof_persona.npz", **{f"{k}_{i}_{n}": v for k, parts in oof.items() for i, (yy, pp) in enumerate(parts)
                                       for n, v in (("y", yy), ("p", pp))})

        # 2. confidence threshold on out-of-fold named-person predictions
        print("\n=== precision of named predictions by confidence threshold (out-of-fold) ===")
        print(f"{'tau':>5s} | {'time: precision':>16s} {'coverage':>9s} {'n':>4s} | {'group: precision':>17s} {'coverage':>9s} {'n':>4s}")
        for tau in np.arange(0.30, 0.96, 0.05):
            cells = []
            for proto in ("time", "group"):
                yt = np.concatenate([yy for yy, _ in oof[proto]])
                pp = np.concatenate([p for _, p in oof[proto]])
                pred, conf = pp.argmax(1), pp.max(1)
                sel = np.isin(pred, named) & (conf >= tau)
                nn_ = int(sel.sum())
                prec = (pred[sel] == yt[sel]).mean() if nn_ else float("nan")
                cells.append((prec, nn_ / max(1, int(np.isin(yt, named).sum())), nn_))
            print(f"{tau:5.2f} | {cells[0][0] * 100:15.1f}% {cells[0][1] * 100:8.1f}% {cells[0][2]:4d} | "
                  f"{cells[1][0] * 100:16.1f}% {cells[1][1] * 100:8.1f}% {cells[1][2]:4d}")
            if chosen is None and cells[0][2] >= 25 and cells[0][0] >= a.precision and cells[1][0] >= a.precision:
                chosen = float(round(tau, 2))
        print(f"\nsuggested threshold for >= {a.precision:.0%} precision on both protocols: {chosen}")

    # 3. final model on every human label
    t0 = time.time()
    net = fit(imgs, y, n_cls, a.epochs, 0, a.model)
    state = {k: v.half().cpu() for k, v in net.state_dict().items()}
    arch, kw, _, _, _ = ARCH[a.model]
    cfg = timm.data.resolve_data_config({}, model=net[0])
    arch_kwargs = dict(kw)
    arch_kwargs.setdefault("img_size", cfg["input_size"][1])
    torch.save({"arch": arch, "arch_kwargs": arch_kwargs, "classes": classes,
                "mean": cfg["mean"], "std": cfg["std"], "input_size": cfg["input_size"][1:], "state": state,
                "meta": {"trained": datetime.now(timezone.utc).isoformat(), "n_train": len(H), "epochs": a.epochs,
                         "model": a.model, "suggested_threshold": chosen, "per_class": dict(cnt)}}, a.out)
    print(f"\nfinal model trained on {len(H)} crops in {time.time() - t0:.0f}s -> {a.out} "
          f"({Path(a.out).stat().st_size / 1e6:.0f} MB)")
    json.dump({"threshold": chosen, "classes": classes}, open("train_final.json", "w"))
    Path("train_final.done").touch()


if __name__ == "__main__":
    main()
