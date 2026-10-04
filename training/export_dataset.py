"""Export labelled crops to folders + manifest.csv for training on another machine.

Run from the project root:  .venv/bin/python training/export_dataset.py [out_dir]
Layout:  out/<kind>/<label>/<file>.jpg   (identities)   and   out/<kind>/_other/<label>/...  (altro, sconosciuto...)
"""
import csv
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "export"
# labels that are not an individual identity: kept aside, not used as classes
NOT_IDENTITY = {"altro", "Gatto sconosciuto", "Persona conosciuta", "Persona sconosciuta"}
MIN_PER_CLASS = 8  # classes with fewer examples are set aside too (too few to learn or validate)

db = sqlite3.connect(ROOT / "data" / "events.db")
rows = db.execute("SELECT id, clip_id, camera, created_at, kind, label, file FROM crops WHERE label IS NOT NULL").fetchall()
counts = {}
for r in rows:
    counts[(r[4], r[5])] = counts.get((r[4], r[5]), 0) + 1

if OUT.exists():
    shutil.rmtree(OUT)
with open(OUT.parent / f"{OUT.name}_manifest.csv", "w", newline="") as mf:
    w = csv.writer(mf)
    w.writerow(["path", "kind", "label", "identity", "clip_id", "camera", "created_at"])
    for cid, clip, camera, created, kind, label, file in rows:
        identity = label not in NOT_IDENTITY and counts[(kind, label)] >= MIN_PER_CLASS
        rel = Path(kind) / (label if identity else f"_other/{label}") / f"{cid}_{file}"
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "data" / "crops" / file, OUT / rel)
        w.writerow([rel.as_posix(), kind, label, int(identity), clip, camera.strip(), created])

for (kind, label), n in sorted(counts.items()):
    ident = label not in NOT_IDENTITY and n >= MIN_PER_CLASS
    print(f"{kind:8s} {label:20s} {n:3d} {'classe' if ident else 'a parte'}")
print("esportato in", OUT)
