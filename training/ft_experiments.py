"""Fine-tuning recipes for CLIP-L person identification, compared on the same folds (human labels only).

Recipes (ideas from docs/research_notes.md):
  base        current recipe: whole network, lr 1e-5, class-weighted CE with label smoothing
  lpft        LP-FT (Kumar et al., ICLR'22): train the head on the frozen backbone first, then fine-tune everything
  lpft_llrd   + layer-wise lr decay (lower layers move less)
  lpft_aug    + harder augmentation for low-res / night-IR crops (down-up-scaling, blur, RandAugment, erasing)
  full        lpft + llrd + aug
  wise_*      WiSE-FT (Wortsman et al., CVPR'22): backbone weights interpolated halfway between the pretrained
              ones and the fine-tuned ones, evaluated from the same run (no extra training)

Usage:  python ft_experiments.py --data export5 [--recipes base,full] [--group 3] [--time 3] [--epochs 15]
Prints balanced accuracy / accuracy per protocol; results go to results_ft_experiments.json.
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
from benchmark_persons import ARCH, DS, transforms
from finetune import group_split, pad_square, time_split

dev = "cuda"
MODEL = "clip_L"
RECIPES = {  # name: (lp_epochs, llrd, strong_aug)
    "base": (0, 1.0, False),
    "lpft": (4, 1.0, False),
    "lpft_llrd": (4, 0.85, False),
    "lpft_aug": (4, 1.0, True),
    "full": (4, 0.85, True),
}


class DownUp:
    """Simulate small / blurry crops: shrink to a random size, scale back up."""
    def __init__(self, lo=40, hi=160, p=0.5):
        self.lo, self.hi, self.p = lo, hi, p

    def __call__(self, im):
        if torch.rand(1) > self.p:
            return im
        s = int(torch.randint(self.lo, self.hi, (1,)))
        return im.resize((s, s), Image.BILINEAR).resize(im.size, Image.BICUBIC)


def strong_transform(mean, std):
    return T.Compose([T.Lambda(pad_square), DownUp(), T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
                      T.RandomHorizontalFlip(), T.RandomApply([T.RandAugment(num_ops=2, magnitude=7)], p=0.6),
                      T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.3), T.RandomGrayscale(0.3),
                      T.ToTensor(), T.Normalize(mean, std), T.RandomErasing(p=0.25, scale=(0.02, 0.15))])


def layer_groups(bb, base_lr, decay):
    """Parameter groups with lr = base_lr * decay^(depth_from_top); depth 0 = embeddings, n+1 = final norm."""
    n = len(bb.blocks)
    groups = {}
    for name, p in bb.named_parameters():
        if name.startswith("blocks."):
            lid = int(name.split(".")[1]) + 1
        elif name.startswith(("cls_token", "pos_embed", "patch_embed", "norm_pre", "reg_token")):
            lid = 0
        else:
            lid = n + 1
        groups.setdefault(lid, []).append(p)
    return [{"params": ps, "lr": base_lr * decay ** (n + 1 - lid)} for lid, ps in groups.items()]


@torch.no_grad()
def predict(net, imgs, ev_tf):
    net.eval()
    out = []
    for i in range(0, len(imgs), 32):
        x = torch.stack([ev_tf(im) for im in imgs[i:i + 32]]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(net(x).argmax(1).cpu())
    return torch.cat(out).numpy()


def train(recipe, imgs, y, n_cls, epochs, seed):
    lp_epochs, decay, strong = RECIPES[recipe]
    arch, kw, (mean, std), bs, ckpt = ARCH[MODEL]
    ev_tf, tr_tf = transforms(mean, std)
    if strong:
        tr_tf = strong_transform(mean, std)
    torch.manual_seed(seed)
    bb = timm.create_model(arch, pretrained=True, num_classes=0, **kw)
    bb.set_grad_checkpointing(ckpt)
    pretrained = copy.deepcopy(bb.state_dict())
    head = nn.Linear(bb.num_features, n_cls)
    net = nn.Sequential(bb, head).to(dev)
    counts = np.bincount(y, minlength=n_cls)
    w = torch.tensor(len(y) / (n_cls * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    dl = torch.utils.data.DataLoader(DS(imgs, y, tr_tf), batch_size=bs, shuffle=True, num_workers=4)

    def run(opt, sched, n_epochs, freeze_bb):
        for _ in range(n_epochs):
            net.train()
            for x, t in dl:
                x, t = x.to(dev), t.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if freeze_bb:
                        with torch.no_grad():
                            f = bb(x)
                        loss = loss_fn(head(f.float()).float(), t)
                    else:
                        loss = loss_fn(net(x).float(), t)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()

    steps_per_epoch = (len(imgs) + bs - 1) // bs
    if lp_epochs:  # stage 1: linear probe
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
        run(opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, lp_epochs * steps_per_epoch), lp_epochs, True)
    top_lr = 1e-5 if decay == 1.0 else 2e-5  # with decay the top layers get the larger lr, the bottom ones far less
    groups = layer_groups(bb, top_lr, decay) + [{"params": head.parameters(), "lr": 1e-3 if not lp_epochs else 3e-4}]
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * steps_per_epoch,
                                                pct_start=0.15)
    run(opt, sched, epochs, False)
    return net, bb, pretrained, ev_tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--recipes", default=",".join(RECIPES))
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--time", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    data = Path(a.data)
    rows = [r for r in csv.DictReader(open(data.parent / f"{data.name}_manifest.csv")) if r["kind"] == "persona"]
    cnt = Counter(r["label"] for r in rows if r["label"])
    classes = sorted(c for c, n in cnt.items() if n >= MIN_CLASS)
    H = [r for r in rows if r["label"] in classes]
    imgs = [Image.open(data / r["path"]).convert("RGB") for r in H]
    y = np.array([classes.index(r["label"]) for r in H])
    clips, times = [r["clip_id"] for r in H], [r["created_at"] for r in H]
    recipes = a.recipes.split(",")
    names = [n for r in recipes for n in (r, "wise_" + r)]
    res = {p: {n: {"bacc": [], "acc": []} for n in names} for p in ("group", "time")}
    print(f"{len(H)} human-labelled crops, {len(classes)} classes", flush=True)
    for proto, s in [("group", i) for i in range(a.group)] + [("time", i) for i in range(a.time)]:
        tr, va = group_split(y, clips, s) if proto == "group" else time_split(clips, times, s)
        tr_imgs, va_imgs = [imgs[i] for i in tr], [imgs[i] for i in va]
        for r in recipes:
            t0 = time.time()
            net, bb, pre, ev_tf = train(r, tr_imgs, y[tr], len(classes), a.epochs, s)
            preds = {r: predict(net, va_imgs, ev_tf)}
            ft = copy.deepcopy(bb.state_dict())  # WiSE-FT: halfway between pretrained and fine-tuned backbone
            bb.load_state_dict({k: (pre[k].to(ft[k].device) + ft[k]) / 2 if ft[k].is_floating_point() else ft[k] for k in ft})
            preds["wise_" + r] = predict(net, va_imgs, ev_tf)
            for n, p in preds.items():
                res[proto][n]["bacc"].append(balanced_accuracy_score(y[va], p))
                res[proto][n]["acc"].append(float((p == y[va]).mean()))
            print(f"  {proto} fold {s} {r}: bacc {res[proto][r]['bacc'][-1] * 100:.1f}%  wise "
                  f"{res[proto]['wise_' + r]['bacc'][-1] * 100:.1f}%  ({time.time() - t0:.0f}s)", flush=True)
            del net, bb, pre
            torch.cuda.empty_cache()
    for proto in res:
        if not res[proto][names[0]]["bacc"]:
            continue
        print(f"\n=== {proto} ===\n{'recipe':16s} {'balanced acc.':>16s} {'accuracy':>14s}")
        for n in names:
            b, c = res[proto][n]["bacc"], res[proto][n]["acc"]
            print(f"{n:16s} {np.mean(b) * 100:8.1f}% ± {np.std(b) * 100:4.1f} {np.mean(c) * 100:8.1f}% ± {np.std(c) * 100:4.1f}")
    json.dump(res, open("results_ft_experiments.json", "w"), indent=1)
    Path("ft_experiments.done").touch()


if __name__ == "__main__":
    main()
