"""Who is it? Logistic regression on crop embeddings, trained on the labels given in the web UI.

Benchmarked on held-out clips: ~90% for the cats, ~65-75% for people (so confidence matters).
"""
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config

SPECIAL = {"altro", "Persona conosciuta", "Persona sconosciuta", "Gatto sconosciuto"}
UNKNOWN = {"Persona sconosciuta", "Gatto sconosciuto"}
_models = {}  # kind -> (signature of the training labels, fitted classifier, examples per label)


def get_classifier(db, kind):
    """Fit on the labelled crops of this kind; refit only when the labels changed."""
    rows = db.execute(
        "SELECT id, label, embedding FROM crops WHERE kind=? AND label IS NOT NULL AND length(embedding)=?",
        (kind, config.EMBED_DIM * 4)).fetchall()
    signature = hash(tuple((r["id"], r["label"]) for r in rows))
    cached = _models.get(kind)
    if cached and cached[0] == signature:
        return cached[1], cached[2]
    labels = [r["label"] for r in rows]
    counts = {lab: labels.count(lab) for lab in set(labels)}
    if len(counts) < 2:  # with a single label every prediction would be that label
        _models[kind] = (signature, None, counts)
        return None, counts
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(X, labels)
    _models[kind] = (signature, clf, counts)
    return clf, counts


def predict(db, kind, embedding):
    """(label, probability) for one embedding, or None while there is not enough labelled data."""
    clf, _ = get_classifier(db, kind)
    if clf is None:
        return None
    p = clf.predict_proba(embedding.reshape(1, -1))[0]
    best = int(p.argmax())
    return str(clf.classes_[best]), float(p[best])


def summarize(categories, preds):
    """Turn the per-crop predictions of one clip into what to show / notify.

    preds: [(kind, label, confidence)] with kind in {"persona", "animale"}.
    Returns [{"kind", "name", "conf", "state"}], state one of
      known      a named person / cat, confident enough
      generic    "Persona conosciuta" (known, but not one of the named people)
      unknown    labelled "Persona/Gatto sconosciuto"
      uncertain  not confident enough to name (or no usable crop at all)
    Crops predicted "altro" (false positives, unusable) are dropped.
    """
    out = []
    for kind in ("persona", "animale"):
        mine = [(label, conf) for k, label, conf in preds if k == kind]
        if kind not in categories:
            continue
        noun = "Persona" if kind == "persona" else "Animale"
        if not mine:  # detected, but no crop big enough to identify
            out.append({"kind": kind, "name": noun, "conf": 0.0, "state": "uncertain"})
            continue
        by = defaultdict(list)
        for label, conf in mine:
            by[label].append(conf)
        # the dominant label is the one with most total confidence; others only if clearly present
        ranked = sorted(by.items(), key=lambda kv: -sum(kv[1]))
        for i, (label, confs) in enumerate(ranked):
            mean = float(np.mean(confs))
            if label == "altro":
                continue
            if i > 0 and not (len(confs) >= 2 and mean >= config.IDENTITY_MIN_CONF[kind]):
                continue
            if mean < config.IDENTITY_MIN_CONF[kind]:
                state, name = "uncertain", ("Persona non riconosciuta" if kind == "persona" else "Gatto non riconosciuto")
            elif label in UNKNOWN:
                state, name = "unknown", label
            elif label == "Persona conosciuta":
                state, name = "generic", label
            else:
                state, name = "known", label
            out.append({"kind": kind, "name": name, "conf": round(mean, 3), "state": state})
    # several "uncertain" entries for one kind are the same unknown thing
    seen, merged = set(), []
    for e in out:
        key = (e["kind"], "uncertain") if e["state"] == "uncertain" else (e["kind"], e["name"])
        if key not in seen:
            seen.add(key)
            merged.append(e)
    return merged
