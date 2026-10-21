"""Store the offline auto-labels in the database (columns auto_*), never touching human labels.

Usage (from the project root):  .venv/bin/python training/import_autolabels.py data/autolabel_persona.csv
"""
import csv
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blinkwatch import store  # noqa: E402  (runs the schema migration)

rows = list(csv.DictReader(open(sys.argv[1])))
db = store.connect()
db.execute("PRAGMA busy_timeout = 20000")
done = skipped = 0
for r in rows:
    cur = db.execute(
        "UPDATE crops SET auto_label=?, auto_conf=?, auto_margin=?, auto_accepted=? WHERE id=? AND label IS NULL",
        (r["auto_label"], float(r["auto_conf"]), float(r["auto_margin"]), int(r["auto_accepted"]), int(r["crop_id"])))
    done += cur.rowcount
    skipped += 1 - cur.rowcount  # already labelled by a human since the export
db.commit()
print(f"imported {done} auto-labels, skipped {skipped} (labelled by hand in the meantime)")
print(db.execute("SELECT auto_label, SUM(auto_accepted), COUNT(*) FROM crops WHERE auto_label IS NOT NULL AND label IS NULL GROUP BY 1").fetchall())
