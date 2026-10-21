"""Auto-label the unconfirmed people crops with the FINE-TUNED CLIP-L, self-training round by round (GPU server).

Human labels are the only ground truth: they set the confidence threshold (out-of-fold probabilities from
train_final.py, human labels only) and they are the only crops ever used for validation. Pseudo-labels
only join the TRAINING side, and pseudo-labelled crops of a validation clip are dropped from that fold.

  round 0: fit on human labels -> score unconfirmed crops -> accept named predictions with conf >= tau
  round k: fit on human + accepted pseudo-labels -> re-score -> re-decide (stops when little changes)
  check  : the same 6 folds as train_final.py (3 grouped by clip, 3 temporal), human + pseudo for training
  final  : fit on human + accepted pseudo-labels, write the checkpoint and an import-ready CSV

Usage:  python autolabel_ft.py --data export5 --oof oof_persona.npz --out persona_clipL_v3.pt
"""
import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score

from autolabel import MIN_CLASS
from benchmark_persons import ARCH, NAMED
from finetune import group_split, time_split
from train_final import MODEL, fit, probabilities

CATCH_ALL = ("altro", "Persona conosciuta", "Persona sconosciuta")  # never auto-accepted: a human decides


def pick_tau(oof_file, named, precision, min_n=25):
    """Lowest threshold whose named predictions reach `precision` on BOTH protocols (human labels only)."""
    z = np.load(oof_file)
    parts = {p: [(z[f"{p}_{i}_y"], z[f"{p}_{i}_p"]) for i in range(3)] for p in ("group", "time")}
    for tau in np.arange(0.30, 0.96, 0.05):
        ok = True
        for p in parts:
            yt = np.concatenate([y for y, _ in parts[p]])
            pp = np.concatenate([q for _, q in parts[p]])
            pred, conf = pp.argmax(1), pp.max(1)
            sel = np.isin(pred, named) & (conf >= tau)
            ok &= sel.sum() >= min_n and (pred[sel] == yt[sel]).mean() >= precision
        if ok:
            return float(round(tau, 2))
    return 0.75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--oof", default="oof_persona.npz")
    ap.add_argument("--out", default="persona_clipL_v3.pt")
    ap.add_argument("--csv", default="autolabel_persona_ft.csv")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--precision", type=float, default=0.95)
    a = ap.parse_args()
    data = Path(a.data)
    rows = [r for r in csv.DictReader(open(data.parent / f"{data.name}_manifest.csv")) if r["kind"] == "persona"]
    cnt = Counter(r["label"] for r in rows if r["label"])
    classes = sorted(c for c, n in cnt.items() if n >= MIN_CLASS)
    n_cls = len(classes)
    named = [classes.index(c) for c in classes if c not in CATCH_ALL]
    H = [i for i, r in enumerate(rows) if r["label"] in classes]
    U = [i for i, r in enumerate(rows) if r["label"] == ""]
    imgs = [Image.open(data / r["path"]).convert("RGB") for r in rows]
    y_h = np.array([classes.index(rows[i]["label"]) for i in H])
    clip = [r["clip_id"] for r in rows]
    when = [r["created_at"] for r in rows]
    tau = pick_tau(a.oof, [classes.index(c) for c in NAMED if c in classes], a.precision)
    print(f"{len(H)} human-labelled, {len(U)} unconfirmed crops; threshold tau={tau} "
          f"(>= {a.precision:.0%} precision on human labels, both protocols)", flush=True)

    def predict_unconfirmed(train_idx, train_y, seed):
        net = fit([imgs[i] for i in train_idx], np.array(train_y), n_cls, a.epochs, seed)
        p = probabilities(net, [imgs[i] for i in U])
        del net
        torch.cuda.empty_cache()
        return p

    # ---- self-training rounds
    t0 = time.time()
    p = predict_unconfirmed(H, y_h, 0)
    acc = np.isin(p.argmax(1), named) & (p.max(1) >= tau)
    print(f"round 0 (human only): accepted {acc.sum()}/{len(U)}  ({time.time() - t0:.0f}s)", flush=True)
    conf_ok, lab = acc, p.argmax(1)          # confident named prediction in the latest round
    stable = acc                              # round 0 has nothing to agree with yet
    for r in range(1, a.rounds + 1):
        t0 = time.time()
        pi = [U[k] for k in np.flatnonzero(stable)]
        p = predict_unconfirmed(H + pi, list(y_h) + list(lab[stable]), r)
        new_conf, new_lab = np.isin(p.argmax(1), named) & (p.max(1) >= tau), p.argmax(1)
        # accept only what was confident in two consecutive rounds with the same label: no flip-flopping
        new_stable = new_conf & conf_ok & (new_lab == lab)
        changed = int((new_stable != stable).sum())
        print(f"round {r}: confident {new_conf.sum()}, stable (2 rounds, same label) {new_stable.sum()}/{len(U)}, "
              f"changed vs previous {changed}  ({time.time() - t0:.0f}s)", flush=True)
        conf_ok, lab, stable, acc = new_conf, new_lab, new_stable, new_stable
        if changed < 10:
            break
    pseudo_idx = [U[k] for k in np.flatnonzero(acc)]
    pseudo_y = lab[acc]
    print("accepted per person:", dict(Counter(classes[l] for l in pseudo_y)), flush=True)

    # ---- import-ready CSV (written now, so a crash during validation loses nothing) (same columns as apply_autolabel.py)
    order = np.argsort(p, axis=1)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop_id", "auto_label", "auto_conf", "auto_margin", "auto_accepted", "second_label"])
        w.writeheader()
        for k, i in enumerate(U):
            top, sec = order[k, -1], order[k, -2]
            w.writerow({"crop_id": rows[i]["crop_id"], "auto_label": classes[top], "auto_conf": round(float(p[k, top]), 3),
                        "auto_margin": round(float(p[k, top] - p[k, sec]), 3), "auto_accepted": int(acc[k]),
                        "second_label": classes[sec]})
    print(f"wrote {a.csv}: {len(pseudo_idx)}/{len(U)} accepted", flush=True)

    # ---- honest check: human + pseudo for training, human only for validation
    print("\n=== validation (human labels only; pseudo-labels train-side only) ===", flush=True)
    res = {"group": [], "time": []}
    for proto, s in [("group", 0), ("group", 1), ("group", 2), ("time", 0), ("time", 1), ("time", 2)]:
        t0 = time.time()
        hy_list = [clip[i] for i in H]
        tr, va = (group_split(y_h, hy_list, s) if proto == "group" else time_split(hy_list, [when[i] for i in H], s))
        val_clips = {hy_list[k] for k in va}
        pseudo_ok = [(i, l) for i, l in zip(pseudo_idx, pseudo_y) if clip[i] not in val_clips]
        tr_idx = [H[k] for k in tr] + [i for i, _ in pseudo_ok]
        tr_y = [y_h[k] for k in tr] + [l for _, l in pseudo_ok]
        net = fit([imgs[i] for i in tr_idx], np.array(tr_y), n_cls, a.epochs, s)
        pv = probabilities(net, [imgs[H[k]] for k in va])
        del net
        torch.cuda.empty_cache()
        b = balanced_accuracy_score(y_h[va], pv.argmax(1)) * 100
        res[proto].append(b)
        print(f"  {proto} fold {s}: train {len(tr)} human + {len(pseudo_ok)} pseudo, val {len(va)}  bacc {b:.1f}%  "
              f"({time.time() - t0:.0f}s)", flush=True)
    for k, v in res.items():
        print(f"  {k}: mean {np.mean(v):.1f}%")

    # ---- final model
    t0 = time.time()
    net = fit([imgs[i] for i in H + pseudo_idx], np.array(list(y_h) + list(pseudo_y)), n_cls, a.epochs, 0)
    arch, kw, (mean, std), _, _ = ARCH[MODEL]
    torch.save({"arch": arch, "arch_kwargs": kw, "classes": classes, "mean": mean, "std": std,
                "state": {k: v.half().cpu() for k, v in net.state_dict().items()},
                "meta": {"trained": datetime.now(timezone.utc).isoformat(), "n_human": len(H), "n_pseudo": len(pseudo_idx),
                         "epochs": a.epochs, "tau": tau, "per_class": dict(cnt)}}, a.out)
    print(f"\nfinal model on {len(H)} human + {len(pseudo_idx)} pseudo crops -> {a.out} ({time.time() - t0:.0f}s)")

    json.dump({"tau": tau, "accepted": len(pseudo_idx), "unconfirmed": len(U), "validation": res}, open("autolabel_ft.json", "w"))
    Path("autolabel_ft.done").touch()


if __name__ == "__main__":
    main()
