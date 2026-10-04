"""Web UI to browse classified clips.

Run:  .venv/bin/uvicorn blinkwatch.web:app --host 0.0.0.0 --port 8765
"""
import json
from pathlib import Path

import numpy as np

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from . import config, store

app = FastAPI(title="Blinkwatch")
PAGE = Path(__file__).with_name("index.html")


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.read_text()


@app.get("/api/events")
def events(category: str = "", camera: str = "", status: str = "", limit: int = 60, before: str = ""):
    sql, args = "SELECT * FROM clips WHERE 1=1", []
    if category == "nessuna":
        sql += " AND status='classified' AND categories=''"
    elif category:
        sql += " AND (',' || categories || ',') LIKE ?"
        args.append(f"%,{category},%")
    if camera:
        sql += " AND camera=?"
        args.append(camera)
    if status:
        sql += " AND status=?"
        args.append(status)
    if before:
        sql += " AND created_at < ?"
        args.append(before)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(min(limit, 200))
    db = store.connect()
    try:
        rows = db.execute(sql, args).fetchall()
        cameras = [r[0] for r in db.execute("SELECT DISTINCT camera FROM clips ORDER BY camera")]
        counts = {r[0] or "nessuna": r[1] for r in db.execute(
            "SELECT categories, COUNT(*) FROM clips WHERE status='classified' GROUP BY categories")}
    finally:
        db.close()
    return {
        "cameras": cameras,
        "total_classified": sum(counts.values()),
        "events": [
            {"id": r["id"], "camera": r["camera"].strip(), "camera_raw": r["camera"], "created_at": r["created_at"],
             "status": r["status"], "has_video": Path(r["path"]).exists(), "categories": [c for c in r["categories"].split(",") if c],
             "details": json.loads(r["details"] or "{}")}
            for r in rows
        ],
    }


@app.get("/video/{clip_id}")
def video(clip_id: str):
    db = store.connect()
    try:
        row = db.execute("SELECT path FROM clips WHERE id=?", (clip_id,)).fetchone()
    finally:
        db.close()
    if not row or not Path(row["path"]).exists():
        raise HTTPException(404)
    return FileResponse(row["path"], media_type="video/mp4")


@app.get("/thumb/{clip_id}")
def thumb(clip_id: str):
    path = config.THUMB_DIR / f"{clip_id}.jpg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


SUGGEST_TOPK = 3  # a label's score = mean of its top-k similarities
PROVISIONAL_BELOW = 10  # suggestions backed by fewer labelled examples are marked as unreliable


def suggest(db, kind, rows):
    """Suggest a label for unlabelled crops by similarity to the labelled ones of the same kind."""
    labelled = db.execute(
        "SELECT label, embedding FROM crops WHERE kind=? AND label IS NOT NULL AND embedding IS NOT NULL", (kind,)
    ).fetchall()
    by_label = {}
    for r in labelled:
        by_label.setdefault(r["label"], []).append(np.frombuffer(r["embedding"], dtype=np.float32))
    if len(by_label) < 2:  # with a single label every suggestion would be that label
        return {}
    mats = {lab: np.stack(v) for lab, v in by_label.items()}
    out = {}
    for r in rows:
        if r["embedding"] is None:
            continue
        e = np.frombuffer(r["embedding"], dtype=np.float32)
        scores = {lab: float(np.sort(m @ e)[::-1][:SUGGEST_TOPK].mean()) for lab, m in mats.items()}
        best = max(scores, key=scores.get)
        out[r["id"]] = {"label": best, "score": round(scores[best], 3), "n": len(mats[best]),
                        "provisional": len(mats[best]) < PROVISIONAL_BELOW}
    return out


@app.get("/label", response_class=HTMLResponse)
def label_page():
    return Path(__file__).with_name("label.html").read_text()


@app.get("/api/crops")
def crops(kind: str = "animale", label: str = "", limit: int = 20):
    """label='' -> unlabelled crops (newest first); otherwise crops with that label."""
    if kind not in config.LABELS:
        raise HTTPException(400, "unknown kind")
    db = store.connect()
    try:
        if label:
            rows = db.execute("SELECT * FROM crops WHERE kind=? AND label=? ORDER BY id DESC LIMIT ?",
                              (kind, label, limit)).fetchall()
        else:
            rows = db.execute("SELECT * FROM crops WHERE kind=? AND label IS NULL ORDER BY id DESC LIMIT ?",
                              (kind, limit)).fetchall()
        stats = {r[0] or "_da_etichettare": r[1] for r in db.execute(
            "SELECT label, COUNT(*) FROM crops WHERE kind=? GROUP BY label", (kind,))}
        sugg = {} if label else suggest(db, kind, rows)
    finally:
        db.close()
    crops = [{**{k: r[k] for k in r.keys() if k != "embedding"}, "suggestion": sugg.get(r["id"])} for r in rows]
    return {"labels": config.LABELS[kind], "stats": stats, "crops": crops}


@app.get("/crop/{crop_id}")
def crop_image(crop_id: int):
    db = store.connect()
    try:
        row = db.execute("SELECT file FROM crops WHERE id=?", (crop_id,)).fetchone()
    finally:
        db.close()
    if not row or not (config.CROP_DIR / row["file"]).exists():
        raise HTTPException(404)
    return FileResponse(config.CROP_DIR / row["file"], media_type="image/jpeg")


@app.post("/api/crops/{crop_id}/label")
def set_label(crop_id: int, label: str | None = Body(None, embed=True)):
    """label=None clears the label (undo)."""
    db = store.connect()
    try:
        row = db.execute("SELECT kind FROM crops WHERE id=?", (crop_id,)).fetchone()
        if row is None:
            raise HTTPException(404)
        if label is not None and label not in config.LABELS[row["kind"]]:
            raise HTTPException(400, "unknown label")
        db.execute("UPDATE crops SET label=? WHERE id=?", (label, crop_id))
        db.commit()
    finally:
        db.close()
    return {"ok": True}
