"""Poll the Blink sync module, download new local-storage clips and classify them.

Run:  .venv/bin/python -m blinkwatch.watcher
"""
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta

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
    await sync.refresh()
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
        try:
            await download(item, blink, path)
        except Exception:
            log.exception("download failed for clip %s", item.id)
            continue
        store.add_clip(db, item.id, item.name, item.created_at, path)
        log.info("downloaded %s", path.name)
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
            guess = await asyncio.to_thread(identify.predict, db, kind, emb)
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
        while True:
            try:
                await blink.refresh()
                await fetch_new_clips(blink, db)
                await classify_pending(classifier, embedder, db)
                delete_old_videos(db)
            except Exception:
                log.exception("loop error")
            await asyncio.sleep(config.POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
