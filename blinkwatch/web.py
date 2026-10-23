"""Web UI to browse classified clips.

Run:  .venv/bin/uvicorn blinkwatch.web:app --host 0.0.0.0 --port 8765
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from . import config, identify, power, store

app = FastAPI(title="Blinkwatch")
app.include_router(power.router)
PAGE = Path(__file__).with_name("index.html")


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.read_text()


@app.get("/api/events")
def events(category: str = "", camera: str = "", status: str = "", limit: int = 60, before: str = "", who: str = "", after: str = ""):
    sql, args = "SELECT * FROM clips WHERE 1=1", []
    if who:  # only clips where this person/cat was identified (state known or uncertain guess)
        sql += " AND identities LIKE ?"
        args.append('%' + json.dumps({"name": who}, ensure_ascii=False)[1:-1] + '%')
    if after:
        sql += " AND created_at > ?"
        args.append(after)
    if category == "nessuna":
        sql += " AND status='classified' AND categories=''"
    elif category:
        sql += " AND (',' || categories || ',') LIKE ?"
        args.append(f"%,{category},%")
    if camera:
        sql += " AND TRIM(camera)=?"  # Blink pads some names with a trailing space
        args.append(camera.strip())
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
        cameras = [r[0] for r in db.execute("SELECT DISTINCT TRIM(camera) AS c FROM clips ORDER BY c")]
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
             "details": json.loads(r["details"] or "{}"), "identities": json.loads(r["identities"] or "[]")}
            for r in rows
        ],
    }


@app.get("/api/summary")
def summary(since: str, days: int = 7):
    """Counts for the home page: clips per category from `since` (local midnight of the oldest day, ISO) in daily buckets,
    who was last seen (identified with state 'known') and when each camera last saw something."""
    start = datetime.fromisoformat(since.replace("Z", "+00:00"))
    db = store.connect()
    try:
        rows = db.execute("SELECT created_at, camera, categories, identities FROM clips WHERE created_at >= ? ORDER BY created_at DESC",
                          (start.isoformat(),)).fetchall()
    finally:
        db.close()
    per_day = [{"person": 0, "animal": 0, "vehicle": 0, "total": 0} for _ in range(max(1, min(days, 30)))]
    seen, cams = {}, {}
    for r in rows:
        at = datetime.fromisoformat(r["created_at"])
        i = int((at - start).total_seconds() // 86400)
        cats = {c for c in r["categories"].split(",") if c}
        if 0 <= i < len(per_day):
            d = per_day[i]
            d["total"] += 1
            d["person"] += "persona" in cats
            d["animal"] += "animale" in cats
            d["vehicle"] += "veicolo" in cats
        cam = r["camera"].strip()
        c = cams.setdefault(cam, {"camera": cam, "last_at": r["created_at"], "count": 0})
        c["count"] += 1
        for ident in json.loads(r["identities"] or "[]"):
            if ident.get("state") == "known" and ident["name"] not in seen:  # rows are newest first
                seen[ident["name"]] = {"name": ident["name"], "kind": ident.get("kind"), "at": r["created_at"], "camera": cam, "conf": ident.get("conf")}
    return {"days": per_day, "seen": sorted(seen.values(), key=lambda x: x["at"], reverse=True),
            "cameras": sorted(cams.values(), key=lambda x: x["camera"]),
            "names": {"persone": config.PERSON_NAMES, "animali": config.CAT_NAMES}}


CAMERAS_PAGE = Path(__file__).with_name("cameras.html")
STORAGE_LOG = config.DATA_DIR / "storage_status.jsonl"


@app.get("/cameras", response_class=HTMLResponse)
def cameras_page():
    return HTMLResponse(CAMERAS_PAGE.read_text(), headers={"Cache-Control": "no-store"})


def _local_midnight(now: datetime, tz_min: int) -> datetime:
    """UTC instant of the last midnight in the client's timezone (tz_min = minutes east of UTC)."""
    local = now + timedelta(minutes=tz_min)
    return local.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(minutes=tz_min)


def _camera_state(age_s: float, expected_s: float) -> str:
    """ok / quiet / stalled from how long a camera has been silent compared with how often it normally records."""
    if age_s <= max(4 * 3600, 10 * expected_s):
        return "ok"
    return "quiet" if age_s <= max(24 * 3600, 30 * expected_s) else "stalled"


@app.get("/api/cameras")
def cameras(tz: int = 0):
    """Per camera: last clip, activity of the last 7 days (per day and per hour of the day), status, and what Blink reports
    (battery, Wi-Fi) when the watcher has recorded it. Plus the USB-storage state of each sync module."""
    now = datetime.now(timezone.utc)
    start = _local_midnight(now, tz) - timedelta(days=6)
    db = store.connect()
    try:
        rows = db.execute("SELECT id, camera, created_at, categories, identities FROM clips WHERE created_at >= ? ORDER BY created_at DESC",
                          (start.isoformat(),)).fetchall()
        last = {}
        for r in db.execute("SELECT camera, MAX(created_at) AS at, COUNT(*) AS n FROM clips GROUP BY camera"):
            last.setdefault(r["camera"].strip(), {"at": r["at"], "n": 0})
            last[r["camera"].strip()]["at"] = max(last[r["camera"].strip()]["at"], r["at"])
            last[r["camera"].strip()]["n"] += r["n"]
        newest = {}
        for r in db.execute("SELECT id, camera, MAX(created_at) AS at FROM clips GROUP BY camera"):
            k = r["camera"].strip()
            if k not in newest or r["at"] > newest[k][1]:
                newest[k] = (r["id"], r["at"])
        status = {r["camera"]: (json.loads(r["data"] or "{}"), r["updated_at"]) for r in db.execute("SELECT * FROM camera_status")}
        processed = db.execute("SELECT MAX(processed_at) FROM clips").fetchone()[0]
    finally:
        db.close()
    today_start = start + timedelta(days=6)
    out = {k: {"camera": k, "total": v["n"], "last_at": v["at"], "last_id": newest[k][0], "today": 0, "week": 0,
               "days": [0] * 7, "hours": [0] * 24, "cats": {"persona": 0, "animale": 0, "veicolo": 0}, "who": {}}
           for k, v in last.items()}
    for r in rows:
        c = out[r["camera"].strip()]
        at = datetime.fromisoformat(r["created_at"])
        c["week"] += 1
        c["days"][min(6, int((at - start).total_seconds() // 86400))] += 1
        c["hours"][(at + timedelta(minutes=tz)).hour] += 1
        c["today"] += at >= today_start
        for cat in r["categories"].split(","):
            if cat in c["cats"]:
                c["cats"][cat] += 1
        for ident in json.loads(r["identities"] or "[]"):
            if ident.get("state") == "known":
                c["who"][ident["name"]] = c["who"].get(ident["name"], 0) + 1
    for k, c in out.items():
        age = (now - datetime.fromisoformat(c["last_at"])).total_seconds()
        expected = 7 * 86400 / c["week"] if c["week"] else 7 * 86400
        c["age_s"] = int(age)
        c["state"] = _camera_state(age, expected)
        c["expected_gap_s"] = int(expected)
        c["blink"], c["blink_at"] = status.get(k, (None, None))
    modules = {}
    if STORAGE_LOG.exists():
        for line in STORAGE_LOG.read_text().splitlines():
            try:
                rec = json.loads(line)
                modules[rec["module"]] = {"module": rec["module"], "status": rec["to"], "since": rec["time"], "cameras": rec["cameras"]}
            except (ValueError, KeyError):
                pass
    return {"now": now.isoformat(), "cameras": sorted(out.values(), key=lambda c: c["camera"]), "modules": list(modules.values()),
            "last_processed": processed}


@app.get("/api/animals")
def animals(tz: int = 0):
    """Each pet: when and where it was last seen, sightings today and over 7 days, and a recent crop as its picture."""
    now = datetime.now(timezone.utc)
    start = _local_midnight(now, tz) - timedelta(days=6)
    today_start = start + timedelta(days=6)
    names = [n for n in config.CAT_NAMES if "sconosciut" not in n.lower()]
    db = store.connect()
    try:
        rows = db.execute("SELECT id, camera, created_at, identities FROM clips WHERE created_at >= ? AND identities LIKE '%animale%' ORDER BY created_at DESC",
                          (start.isoformat(),)).fetchall()
        res = {n: {"name": n, "last_at": None, "camera": None, "clip_id": None, "today": 0, "week": 0, "days": [0] * 7, "where": {}} for n in names}
        for r in rows:
            at = datetime.fromisoformat(r["created_at"])
            for ident in json.loads(r["identities"] or "[]"):
                a = res.get(ident.get("name"))
                if a is None or ident.get("state") != "known":
                    continue
                if a["last_at"] is None:
                    a.update(last_at=r["created_at"], camera=r["camera"].strip(), clip_id=r["id"])
                a["week"] += 1
                a["today"] += at >= today_start
                a["days"][min(6, int((at - start).total_seconds() // 86400))] += 1
                a["where"][r["camera"].strip()] = a["where"].get(r["camera"].strip(), 0) + 1
        for n, a in res.items():
            crop = db.execute("SELECT id FROM crops WHERE kind='animale' AND (label=? OR (label IS NULL AND pred_label=? AND pred_conf>=0.6)) "
                              "ORDER BY created_at DESC LIMIT 1", (n, n)).fetchone()
            a["crop_id"] = crop["id"] if crop else None
            if a["last_at"] is None:  # not seen this week: look further back for the last time
                old = db.execute("SELECT id, camera, created_at FROM clips WHERE identities LIKE ? ORDER BY created_at DESC LIMIT 1",
                                 ("%" + json.dumps({"name": n}, ensure_ascii=False)[1:-1] + "%",)).fetchone()
                if old:
                    a.update(last_at=old["created_at"], camera=old["camera"].strip(), clip_id=old["id"])
    finally:
        db.close()
    return {"animals": list(res.values())}


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


PROVISIONAL_BELOW = 10  # suggestions backed by fewer labelled examples are marked as unreliable
def suggest(db, kind, rows):
    """Suggest a label for unlabelled crops (benchmark: ~90% on cats, ~65-75% on people)."""
    clf, counts = identify.get_classifier(db, kind)
    rows = [r for r in rows if r["embedding"] is not None and len(r["embedding"]) == config.EMBED_DIM * 4]
    if clf is None or not rows:
        return {}
    proba = clf.predict_proba(np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]))
    out = {}
    for r, p in zip(rows, proba):
        best = int(p.argmax())
        label = clf.classes_[best]
        out[r["id"]] = {"label": label, "score": round(float(p[best]), 3), "n": counts[label],
                        "provisional": counts[label] < PROVISIONAL_BELOW}
    return out


@app.get("/label", response_class=HTMLResponse)
def label_page():
    return Path(__file__).with_name("label.html").read_text()


@app.get("/api/crops")
def crops(kind: str = "animale", label: str = "", limit: int = 20):
    """label='' -> crops still to label (the least sure first, then newest);
    label='auto:<name>' -> crops auto-labelled <name> and not yet checked by a human;
    otherwise -> crops a human labelled with that label."""
    if kind not in config.LABELS:
        raise HTTPException(400, "unknown kind")
    db = store.connect()
    try:
        if label.startswith("auto:"):
            rows = db.execute("SELECT * FROM crops WHERE kind=? AND label IS NULL AND auto_accepted=1 AND auto_label=?"
                              " ORDER BY auto_conf ASC LIMIT ?", (kind, label[5:], limit)).fetchall()
        elif label:
            rows = db.execute("SELECT * FROM crops WHERE kind=? AND label=? ORDER BY id DESC LIMIT ?",
                              (kind, label, limit)).fetchall()
        else:  # not labelled and not confidently auto-labelled; unknown margin last
            rows = db.execute("SELECT * FROM crops WHERE kind=? AND label IS NULL AND COALESCE(auto_accepted,0)=0"
                              " ORDER BY COALESCE(pred_margin, auto_margin) IS NULL, COALESCE(pred_margin, auto_margin) ASC, id DESC LIMIT ?", (kind, limit)).fetchall()
        stats = {r[0]: r[1] for r in db.execute(
            "SELECT label, COUNT(*) FROM crops WHERE kind=? AND label IS NOT NULL GROUP BY label", (kind,))}
        stats["_da_etichettare"] = db.execute(
            "SELECT COUNT(*) FROM crops WHERE kind=? AND label IS NULL AND COALESCE(auto_accepted,0)=0", (kind,)).fetchone()[0]
        for name, n in db.execute("SELECT auto_label, COUNT(*) FROM crops WHERE kind=? AND label IS NULL AND auto_accepted=1"
                                  " GROUP BY auto_label", (kind,)):
            stats[f"auto:{name}"] = n
        sugg = {} if (label and not label.startswith("auto:")) else suggest(db, kind, rows)
    finally:
        db.close()
    for r in rows:  # the big models beat the small live one: prefer their answer when there is one
        if r["label"] is None and r["pred_model"]:  # fine-tuned model, saved when the crop was made / re-scored
            sugg[r["id"]] = {"label": r["pred_label"], "score": round(r["pred_conf"], 3), "n": None,
                             "provisional": False, "source": "auto"}
        elif r["label"] is None and r["auto_label"] is not None:
            sugg[r["id"]] = {"label": r["auto_label"], "score": round(r["auto_conf"], 3), "n": None,
                             "provisional": False, "source": "auto"}
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
