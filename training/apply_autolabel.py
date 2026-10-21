"""Re-run the self-training step of autolabel.py at a chosen confidence threshold, from cached features.

Usage:  python apply_autolabel.py <manifest.csv> <features.npy> <tau> <out.csv> [kind]
Human labels are the only ground truth; the output has one row per UNLABELLED crop.
"""
import csv
import sys
from collections import Counter

import numpy as np

from autolabel import MIN_CLASS, self_train

manifest, feats, tau, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
kind = sys.argv[5] if len(sys.argv) > 5 else "persona"
rows = [r for r in csv.DictReader(open(manifest)) if r["kind"] == kind]
X = np.load(feats)
assert len(X) == len(rows), "features and manifest do not match"
labels = [r["label"] for r in rows]
cnt = Counter(l for l in labels if l)
classes = sorted(c for c, n in cnt.items() if n >= MIN_CLASS)
H = [i for i, l in enumerate(labels) if l in classes]
U = [i for i, l in enumerate(labels) if l == ""]
yh = np.array([classes.index(labels[i]) for i in H])
# never auto-accept the catch-all labels: strangers / known-but-unnamed / unusable crops go to a human
ok = {classes.index(c) for c in classes if c not in ("altro", "Persona conosciuta", "Persona sconosciuta", "Gatto sconosciuto")}
clf, accepted, lab, conf, proba, hist = self_train(X[H], yh, X[U], tau, ok, C=10)
order = np.argsort(proba, axis=1)
second, second_conf = clf.classes_[order[:, -2]], np.take_along_axis(proba, order[:, -2:-1], axis=1)[:, 0]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["crop_id", "auto_label", "auto_conf", "auto_margin", "auto_accepted", "second_label"])
    w.writeheader()
    for k, i in enumerate(U):
        w.writerow({"crop_id": rows[i]["crop_id"], "auto_label": classes[lab[k]], "auto_conf": round(float(conf[k]), 3),
                    "auto_margin": round(float(conf[k] - second_conf[k]), 3), "auto_accepted": int(accepted[k]),
                    "second_label": classes[second[k]]})
print(f"tau {tau}: accepted {int(accepted.sum())}/{len(U)} in {len(hist)} rounds:",
      dict(Counter(classes[l] for l, a in zip(lab, accepted) if a)))
print("left for a human:", len(U) - int(accepted.sum()), "| predicted as:", dict(Counter(classes[l] for l, a in zip(lab, accepted) if not a)))
