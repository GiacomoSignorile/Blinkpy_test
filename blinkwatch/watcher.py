"""Poll the Blink sync module, download new local-storage clips and classify them.

Run:  .venv/bin/python -m blinkwatch.watcher
"""
import asyncio
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import cv2
from aiohttp import ClientSession

from . import config, store
from .blink_client import connect, download
from .classifier import ClipClassifier, fallback_thumb
from .embedder import Embedder
from . import identify, notify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("watcher")


async def fetch_new_clips(blink, db):
    for name in config.SYNC_MODULE_NAMES:
        sync = blink.sync.get(name)
        if sync is None:
            log.error("Sync module %r not found (available: %s)", name, list(blink.sync))
            continue
        await fetch_from_sync(blink, db, sync)


async def fetch_from_sync(blink, db, sync):
    # no sync.refresh() here: blink.refresh() just refreshed every sync module (and its clip list)
    if not (sync.local_storage and sync.local_storage_manifest_ready):
        log.warning("Manifest not ready for %s", sync.name)
        return
    done = 0
    for item in sorted(sync._local_storage["manifest"], key=lambda i: i.created_at, reverse=True):
        if done >= config.MAX_CLIPS_PER_POLL:
            break
        if store.has_clip(db, item.id):
            continue
        age = datetime.now(item.created_at.tzinfo) - item.created_at
        if age > timedelta(hours=config.MAX_CLIP_AGE_HOURS):
            break  # sorted newest-first: everything after is older
        path = config.DOWNLOAD_PATH / f"{item.name}_{item.id}_{item.created_at:%Y%m%d_%H%M%S}.mp4"
        seen_after = age.total_seconds()
        t0 = time.monotonic()
        try:
            await download(item, blink, path)
        except Exception:
            log.exception("download failed for clip %s", item.id)
            continue
        store.add_clip(db, item.id, item.name, item.created_at, path)
        log.info("downloaded %s  [timing: in manifest %.0fs after recording, download took %.0fs]",
                 path.name, seen_after, time.monotonic() - t0)
        done += 1
        await asyncio.sleep(2)  # be gentle with the API


async def classify_pending(classifier, embedder, db):
    for row in store.pending(db):
        try:
            cats, details, crops = await asyncio.to_thread(classifier.classify, row["path"], config.THUMB_DIR / f"{row['id']}.jpg")
        except Exception:
            log.exception("classification failed for %s", row["path"])
            store.set_result(db, row["id"], [], {}, status="error")
            continue
        preds = []  # (kind, label, confidence) of each crop, to say who is in the clip
        for n, (kind, name, conf, image) in enumerate(crops):
            rel = f"{row['id']}_{n}.jpg"
            config.CROP_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(config.CROP_DIR / rel), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            emb = await asyncio.to_thread(embedder.embed, image)
            try:  # same thread as the DB connection (sqlite objects cannot cross threads)
                guess = identify.predict(db, kind, emb)
            except Exception:
                log.exception("identification failed, saving the crop without a name")
                guess = None
            if guess:
                preds.append((kind, *guess))
            store.add_crop(db, row["id"], row["camera"], row["created_at"], name, round(conf, 3), rel,
                           kind=kind, embedding=emb.tobytes(),
                           pred_label=guess[0] if guess else None, pred_conf=round(guess[1], 3) if guess else None)
        identities = identify.summarize(cats, preds)
        store.set_result(db, row["id"], cats, details, identities=identities)
        try:
            await asyncio.to_thread(notify.send_for_clip, row, cats, details, identities)
        except Exception:
            log.exception("notification failed")
        log.info("[%s] %s -> %s %s %s", row["camera"], row["created_at"], sorted(cats) or "nulla", details,
                 [f"{i['name']} {i['conf']:.0%}" for i in identities])


def delete_old_videos(db):
    """Delete classified videos past the retention window, keeping preview + classification."""
    cutoff = datetime.now().astimezone() - timedelta(hours=config.RETENTION_HOURS)
    for row in store.finished(db):
        video = Path(row["path"])
        if not video.exists() or datetime.fromisoformat(row["created_at"]) > cutoff:
            continue
        thumb = config.THUMB_DIR / f"{row['id']}.jpg"
        if not thumb.exists():
            fallback_thumb(video, thumb)
        video.unlink()
        log.info("deleted old video %s", video.name)


_storage_seen = {}
STORAGE_LOG = config.DATA_DIR / "storage_status.jsonl"  # history of USB-storage state changes, survives restarts


def _last_known_storage():
    known = {}
    if STORAGE_LOG.exists():
        for line in STORAGE_LOG.read_text().splitlines():
            try:
                rec = json.loads(line)
                known[rec["module"]] = rec["to"]
            except (ValueError, KeyError):
                pass
    return known


_known_status = _last_known_storage()


def log_storage_status(blink):
    """Follow each sync module's USB-storage state as Blink reports it, and log every change.

    blinkpy reads the state only once at startup, so a drive that breaks (or is fixed) while we run would go
    unnoticed. Here the state is re-read from the home screen after each refresh and pushed back into blinkpy,
    so downloads resume by themselves once Blink says the storage is "active" again.
    """
    sensitive = ("id", "serial", "mac", "ip", "ssid", "token", "key", "address", "lat", "lon", "network")
    homescreen = {m.get("id"): m for m in (blink.homescreen or {}).get("sync_modules", [])}
    for name, sync in blink.sync.items():
        mod = homescreen.get(sync.sync_id) or {}
        if "local_storage_status" in mod:
            sync._local_storage["status"] = mod["local_storage_status"] == "active"
            sync._local_storage["enabled"] = mod.get("local_storage_enabled")
            sync._local_storage["compatible"] = mod.get("local_storage_compatible")
        state = {
            "cameras": sorted(c.strip() for c in sync.cameras),
            "manifest_stale": sync._local_storage.get("manifest_stale"),
        }
        for k, v in mod.items():  # scalar fields Blink reports for this module (storage, wifi, firmware...)
            if isinstance(v, (str, int, float, bool)) and (k.startswith("local_storage")
                                                           or not any(w in k.lower() for w in sensitive)):
                state[k] = v
        key = {k: v for k, v in state.items() if k not in ("manifest_stale", "last_hb")}  # noisy fields
        if _storage_seen.get(name) != key:
            if name in _storage_seen:
                log.warning("storage status CHANGED [%s]: %s", name, json.dumps(state, ensure_ascii=False, default=str))
            else:
                log.info("storage status [%s]: %s", name, json.dumps(state, ensure_ascii=False, default=str))
            _storage_seen[name] = key
        status = mod.get("local_storage_status")
        if status is not None and status != _known_status.get(name):
            prev = _known_status.get(name)
            with open(STORAGE_LOG, "a") as f:
                f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "module": name, "from": prev,
                                    "to": status, "fw": mod.get("fw_version"), "cameras": state["cameras"]},
                                   ensure_ascii=False) + "\n")
            _known_status[name] = status
            if prev is not None:  # a real change (the first record per module is just the baseline)
                ok = status == "active"
                notify.send_text(f"Archivio USB {name}: {status}", f"Prima: {prev}. Telecamere: {', '.join(state['cameras'])}",
                                 priority=3 if ok else 5, tags=["white_check_mark" if ok else "warning"])


async def cleanup(db):
    delete_old_videos(db)  # sync on purpose: sqlite connections cannot be used from other threads


def backfill_embeddings(embedder, db):
    """Crops with no embedding, or one from an older model (different size)."""
    for row in db.execute("SELECT id, file FROM crops WHERE embedding IS NULL OR length(embedding) != ?",
                          (config.EMBED_DIM * 4,)).fetchall():
        image = cv2.imread(str(config.CROP_DIR / row["file"]))
        if image is not None:
            db.execute("UPDATE crops SET embedding=? WHERE id=?", (embedder.embed(image).tobytes(), row["id"]))
    db.commit()


async def main():
    config.DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    db = store.connect()
    classifier = ClipClassifier()
    embedder = Embedder()
    backfill_embeddings(embedder, db)
    async with ClientSession() as session:
        blink = await connect(session)
        blink.refresh_rate = config.POLL_INTERVAL
        while True:
            # independent steps: a failure in one must not stop the others (e.g. old-video cleanup)
            times = {}
            for name, step in (("refresh", lambda: blink.refresh()), ("fetch", lambda: fetch_new_clips(blink, db)),
                               ("classify", lambda: classify_pending(classifier, embedder, db)),
                               ("cleanup", lambda: cleanup(db))):
                t0 = time.monotonic()
                try:
                    await step()
                except Exception:
                    log.exception("loop step %s failed", name)
                times[name] = time.monotonic() - t0
            try:
                log_storage_status(blink)
            except Exception:
                log.exception("storage status logging failed")
            log.info("loop timing: " + " ".join(f"{k} {v:.1f}s" for k, v in times.items()))
            await asyncio.sleep(config.POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
