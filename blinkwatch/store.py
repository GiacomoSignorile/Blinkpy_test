import json
import sqlite3
from datetime import datetime, timezone

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS crops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id TEXT,
    camera TEXT,
    created_at TEXT,
    det_class TEXT,       -- what YOLO thought it was (cat, bird, dog...)
    conf REAL,
    file TEXT,            -- path relative to CROP_DIR
    label TEXT,           -- NULL = not labelled yet; a name, or 'altro'
    kind TEXT DEFAULT 'animale',  -- animale | persona
    embedding BLOB,       -- float32 feature vector, for label suggestions
    pred_label TEXT,      -- automatic identification at the time the crop was saved
    pred_conf REAL
);
CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,
    camera TEXT,
    created_at TEXT,
    path TEXT,
    status TEXT DEFAULT 'downloaded',  -- downloaded | classified | error
    categories TEXT DEFAULT '',        -- e.g. "persona,veicolo"
    details TEXT DEFAULT '',           -- JSON: per-label max confidence
    processed_at TEXT,
    identities TEXT DEFAULT ''         -- JSON: who/what was recognised, see identify.summarize
);
"""


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    for table, wanted in (("crops", (("kind", "TEXT DEFAULT 'animale'"), ("embedding", "BLOB"),
                                     ("pred_label", "TEXT"), ("pred_conf", "REAL"))),
                          ("clips", (("identities", "TEXT DEFAULT ''"),))):
        cols = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
        for col, ddl in wanted:
            if col not in cols:  # tables created before these columns existed
                db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
    db.commit()
    return db


def has_clip(db, clip_id) -> bool:
    return db.execute("SELECT 1 FROM clips WHERE id=?", (str(clip_id),)).fetchone() is not None


def add_clip(db, clip_id, camera, created_at, path):
    db.execute(
        "INSERT OR IGNORE INTO clips (id, camera, created_at, path) VALUES (?,?,?,?)",
        (str(clip_id), camera, created_at.isoformat(), str(path)),
    )
    db.commit()


def pending(db):
    return db.execute("SELECT * FROM clips WHERE status='downloaded' ORDER BY created_at").fetchall()


def finished(db):
    return db.execute("SELECT * FROM clips WHERE status!='downloaded'").fetchall()


def set_result(db, clip_id, categories, details, status="classified", identities=None):
    db.execute(
        "UPDATE clips SET status=?, categories=?, details=?, processed_at=?, identities=? WHERE id=?",
        (status, ",".join(sorted(categories)), json.dumps(details),
         datetime.now(timezone.utc).isoformat(), json.dumps(identities or [], ensure_ascii=False), str(clip_id)),
    )
    db.commit()


def add_crop(db, clip_id, camera, created_at, det_class, conf, file, kind="animale", embedding=None,
             pred_label=None, pred_conf=None):
    cur = db.execute(
        "INSERT INTO crops (clip_id, camera, created_at, det_class, conf, file, kind, embedding, pred_label, pred_conf)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (str(clip_id), camera, created_at, det_class, conf, file, kind, embedding, pred_label, pred_conf),
    )
    db.commit()
    return cur.lastrowid
