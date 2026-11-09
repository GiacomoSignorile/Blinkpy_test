"""Score the still-unlabelled people crops with the production model (suggestions + review order).

Run from the project root, with the database folder given:  DATA_DIR=data .venv/bin/python training/score_unlabeled.py
Writes pred_label / pred_conf / pred_margin / pred_model only; human labels and auto_* columns are never touched.
"""
import sqlite3
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blinkwatch import config, store  # noqa: E402
from blinkwatch.person_model import PersonModel  # noqa: E402

model = PersonModel.load_if_present()
if model is None:
    sys.exit("no model checkpoint found")
db = store.connect()
db.execute("PRAGMA busy_timeout = 30000")
rows = db.execute("SELECT id, file FROM crops WHERE kind='persona' AND label IS NULL "
                  "AND (pred_model IS NULL OR pred_model != ?)", (model.name,)).fetchall()
print(f"{len(rows)} crops to score with {model.name}", flush=True)


pending = []


def flush():
    """Write the scores in one short transaction: the watcher and the web UI share this database,
    and a transaction held open while the model runs blocks both of them."""
    for _ in range(120):
        try:
            with db:
                db.executemany("UPDATE crops SET pred_label=?, pred_conf=?, pred_margin=?, pred_model=? "
                               "WHERE id=? AND label IS NULL", pending)
            pending.clear()
            return
        except sqlite3.OperationalError:
            time.sleep(5)
    raise SystemExit("database stayed locked for 10 minutes")


t0 = time.time()
for i, r in enumerate(rows, 1):
    img = cv2.imread(str(config.CROP_DIR / r["file"]))
    if img is None:
        continue
    label, prob, margin = model.predict(img)
    pending.append((label, round(prob, 3), round(margin, 3), model.name, r["id"]))
    if i % 100 == 0:
        flush()
        print(f"  {i}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)
flush()
print(f"done in {time.time() - t0:.0f}s")
print(db.execute("SELECT pred_label, COUNT(*) FROM crops WHERE kind='persona' AND label IS NULL AND pred_model=? GROUP BY 1 ORDER BY 2 DESC",
                 (model.name,)).fetchall() and [tuple(x) for x in db.execute(
    "SELECT pred_label, COUNT(*) FROM crops WHERE kind='persona' AND label IS NULL AND pred_model=? GROUP BY 1 ORDER BY 2 DESC", (model.name,))])
