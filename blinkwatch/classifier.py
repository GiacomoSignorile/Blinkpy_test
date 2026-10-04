"""Classify a video clip as containing persone / veicoli / animali using YOLO."""
import math

import cv2
from ultralytics import YOLO

from . import config


class ClipClassifier:
    def __init__(self):
        self.model = YOLO(config.MODEL)

    def classify(self, path: str, thumb_path=None):
        """Return (categories, {label: max_confidence}, animal crops); optionally save a preview JPEG.

        Crops are [(kind, class_name, conf, image)], the best few per kind spaced >= 1s apart.

        The preview is the frame with the most confident detection (boxes drawn),
        or the first sampled frame when nothing was detected.
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"cannot open video {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(fps / config.SAMPLE_FPS))
        details: dict[str, float] = {}
        best_score, best_frame, first_frame = 0.0, None, None
        candidates = []  # (quality, frame_idx, class_name, crop, conf) of animal detections
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    res = self.model(frame, conf=config.CONF_THRESHOLD, verbose=False)[0]
                    if first_frame is None:
                        first_frame = frame
                    for box in res.boxes:
                        name = res.names[int(box.cls)]
                        if name in config.CATEGORIES:
                            conf = float(box.conf)
                            details[name] = max(details.get(name, 0), round(conf, 3))
                            if conf > best_score:
                                best_score, best_frame = conf, res.plot()
                            if config.CATEGORIES[name] in ("animale", "persona"):
                                crop = crop_box(frame, box.xyxy[0].tolist())
                                if crop is not None:
                                    candidates.append((crop_quality(crop, conf), i, name, crop, conf))
                i += 1
        finally:
            cap.release()
        if thumb_path is not None:
            save_thumb(best_frame if best_frame is not None else first_frame, thumb_path)
        return {config.CATEGORIES[n] for n in details}, details, pick_crops(candidates, step)


def save_thumb(frame, thumb_path):
    if frame is None:
        return
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, round(h * 640 / w)))
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(thumb_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])


def fallback_thumb(video_path, thumb_path):
    """Preview from the first frame, for clips classified before previews existed."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
        if ok:
            save_thumb(frame, thumb_path)
    finally:
        cap.release()


def crop_box(frame, xyxy, pad=0.15):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    if min(x2 - x1, y2 - y1) < config.MIN_CROP_SIDE:
        return None
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    return frame[y1:y2, x1:x2].copy()


def crop_quality(crop, conf):
    """Score for how useful a crop is to tell individuals apart: confident, big and sharp."""
    h, w = crop.shape[:2]
    sharpness = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return conf * (w * h) ** 0.5 * math.log1p(sharpness)


def pick_crops(candidates, step):
    """Best-quality crops per kind (animale/persona), at least one second apart (no near-duplicates)."""
    limits = {"animale": config.MAX_CROPS_PER_CLIP, "persona": config.MAX_PERSON_CROPS_PER_CLIP}
    picked = {"animale": [], "persona": []}
    for _, idx, name, crop, conf in sorted(candidates, key=lambda c: -c[0]):
        kind = config.CATEGORIES[name]
        mine = picked[kind]
        if len(mine) < limits[kind] and all(abs(idx - p[0]) >= step * config.SAMPLE_FPS for p in mine):
            mine.append((idx, name, conf, crop))
    return [(kind, name, conf, crop) for kind, items in picked.items() for _, name, conf, crop in items]
