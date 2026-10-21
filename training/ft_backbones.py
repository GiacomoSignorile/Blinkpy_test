"""Fine-tune any timm backbone on human labels with the current recipe, with seed ensembles and a model soup.

For every fold, N seeds are trained from the same pretrained weights and the same head initialisation (only the
data order / augmentation differ), so their weights can be averaged ("model soup", Wortsman et al., ICML'22).
Reported per protocol (balanced accuracy, mean ± std over folds):
  single    average of the individual seeds
  ensemble  mean of the seeds' softmax outputs
  soup      one model with the seeds' weights averaged

Usage:  python ft_backbones.py --data export5 --arch vit_pe_core_large_patch14_336.fb [--seeds 3] [--group 3] [--time 3]
"""
import argparse
import copy
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import balanced_accuracy_score

from autolabel import MIN_CLASS
from benchmark_persons import DS
from finetune import group_split, pad_square, time_split

dev = "cuda"


def make_tfs(cfg):
    size = cfg["input_size"][1:]
    ev = T.Compose([T.Lambda(pad_square), T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    tr = T.Compose([T.Lambda(pad_square), T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.2)),
                    T.RandomHorizontalFlip(), T.RandomApply([T.ColorJitter(0.3, 0.3, 0.2)], p=0.8),
                    T.RandomGrayscale(0.3), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    return ev, tr


def build(arch, n_cls, kw):
    bb = timm.create_model(arch, pretrained=True, num_classes=0, **kw)
    bb.set_grad_checkpointing(True)
    g = torch.Generator().manual_seed(0)  # same head init for every seed -> weights can be averaged
    head = nn.Linear(bb.num_features, n_cls)
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=g) * 0.02)
        head.bias.zero_()
    return bb, nn.Sequential(bb, head).to(dev)


def train(arch, kw, imgs, y, n_cls, epochs, seed, bs=16, lr_b=1e-5):
    bb, net = build(arch, n_cls, kw)
    cfg = timm.data.resolve_data_config({}, model=bb)
    ev_tf, tr_tf = make_tfs(cfg)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW([{"params": bb.parameters(), "lr": lr_b}, {"params": net[1].parameters(), "lr": 1e-3}],
                            weight_decay=0.05)
    steps = epochs * ((len(imgs) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_b, 1e-3], total_steps=steps, pct_start=0.15)
    counts = np.bincount(y, minlength=n_cls)
    w = torch.tensor(len(y) / (n_cls * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
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
    return net, ev_tf


@torch.no_grad()
def probs(net, imgs, ev_tf):
    net.eval()
    out = []
    for i in range(0, len(imgs), 32):
        x = torch.stack([ev_tf(im) for im in imgs[i:i + 32]]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.softmax(net(x).float(), 1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--time", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img-size", type=int, help="override the model's default input size (e.g. 224 for dinov2)")
    a = ap.parse_args()
    kw = {"img_size": a.img_size} if a.img_size else {}
    data = Path(a.data)
    rows = [r for r in csv.DictReader(open(data.parent / f"{data.name}_manifest.csv")) if r["kind"] == "persona"]
    cnt = Counter(r["label"] for r in rows if r["label"])
    classes = sorted(c for c, n in cnt.items() if n >= MIN_CLASS)
    H = [r for r in rows if r["label"] in classes]
    imgs = [Image.open(data / r["path"]).convert("RGB") for r in H]
    y = np.array([classes.index(r["label"]) for r in H])
    clips, times = [r["clip_id"] for r in H], [r["created_at"] for r in H]
    res = {p: {k: [] for k in ("single", "ensemble", "soup")} for p in ("group", "time")}
    print(f"{a.arch}: {len(H)} crops, {len(classes)} classes, {a.seeds} seeds", flush=True)
    for proto, s in [("group", i) for i in range(a.group)] + [("time", i) for i in range(a.time)]:
        t0 = time.time()
        tr, va = group_split(y, clips, s) if proto == "group" else time_split(clips, times, s)
        tr_imgs, va_imgs = [imgs[i] for i in tr], [imgs[i] for i in va]
        ps, states = [], []
        for sd in range(a.seeds):
            net, ev_tf = train(a.arch, kw, tr_imgs, y[tr], len(classes), a.epochs, 100 * s + sd)
            ps.append(probs(net, va_imgs, ev_tf))
            states.append({k: v.detach().float().cpu() for k, v in net.state_dict().items()})
            del net
            torch.cuda.empty_cache()
        _, net = build(a.arch, len(classes), kw)
        net.load_state_dict({k: sum(st[k] for st in states) / len(states) for k in states[0]})
        soup = probs(net, va_imgs, ev_tf)
        del net, states
        torch.cuda.empty_cache()
        bacc = lambda p: balanced_accuracy_score(y[va], p.argmax(1))
        res[proto]["single"].append(float(np.mean([bacc(p) for p in ps])))
        res[proto]["ensemble"].append(bacc(np.mean(ps, 0)))
        res[proto]["soup"].append(bacc(soup))
        print(f"  {proto} fold {s}: single {res[proto]['single'][-1] * 100:.1f}  ensemble {res[proto]['ensemble'][-1] * 100:.1f}"
              f"  soup {res[proto]['soup'][-1] * 100:.1f}  ({time.time() - t0:.0f}s)", flush=True)
    for proto in res:
        if res[proto]["single"]:
            print(f"=== {proto}: " + "  ".join(f"{k} {np.mean(v) * 100:.1f} ± {np.std(v) * 100:.1f}" for k, v in res[proto].items()))
    tag = a.arch.split(".")[0]
    json.dump(res, open(f"results_ft_{tag}.json", "w"), indent=1)
    Path(f"ft_{tag}.done").touch()


if __name__ == "__main__":
    main()
