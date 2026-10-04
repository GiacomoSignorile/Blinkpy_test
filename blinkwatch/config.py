import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path):
    """Minimal .env loader (KEY=VALUE lines); real environment variables win."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_env(ROOT / ".env")
BLINK_USERNAME = os.environ.get("BLINK_USERNAME")
BLINK_PASSWORD = os.environ.get("BLINK_PASSWORD")
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT))  # credentials, db, 2fa file live here
TWOFA_FILE = DATA_DIR / "2fa_code.txt"  # write the 2FA code here when the watcher asks for it
CREDENTIALS_FILE = DATA_DIR / "blink_credentials.json"
DOWNLOAD_PATH = Path(os.environ.get("DOWNLOAD_PATH", "/home/giacomo/Documenti/Recordings"))
DB_PATH = DATA_DIR / "events.db"
THUMB_DIR = DATA_DIR / "thumbs"  # preview images, kept after the video is deleted
CROP_DIR = DATA_DIR / "crops"  # animal crops kept permanently, to label and train on
CAT_NAMES = [n.strip() for n in os.environ.get("CAT_NAMES", "Mingo,Mango,Buccio,Luna,Gatto sconosciuto").split(",")]
PERSON_NAMES = [n.strip() for n in os.environ.get("PERSON_NAMES", "Io,Mamma,Papà,Rossana,Domenico").split(",")]
EMBED_MODEL = "vit_small_patch14_dinov2.lvd142m"
MAX_PERSON_CROPS_PER_CLIP = int(os.environ.get("MAX_PERSON_CROPS_PER_CLIP", 4))
# labels offered for each kind of crop; "altro" = not that kind of thing / unusable
LABELS = {
    "animale": CAT_NAMES + ["altro"],
    "persona": PERSON_NAMES + ["Persona conosciuta", "Persona sconosciuta", "altro"],
}
MAX_CROPS_PER_CLIP = int(os.environ.get("MAX_CROPS_PER_CLIP", 6))
MIN_CROP_SIDE = 40  # px, ignore tiny detections
RETENTION_HOURS = float(os.environ.get("RETENTION_HOURS", 1))  # videos older than this are deleted

SYNC_MODULE_NAMES = [n.strip() for n in os.environ.get("SYNC_MODULE_NAMES", "Casa,Casa online").split(",") if n.strip()]
POLL_INTERVAL = 30  # seconds between manifest refreshes
MAX_CLIP_AGE_HOURS = float(os.environ.get("MAX_CLIP_AGE_HOURS", 1))  # ignore clips older than this
MAX_CLIPS_PER_POLL = int(os.environ.get("MAX_CLIPS_PER_POLL", 5))

# Classifier
MODEL = os.environ.get("YOLO_MODEL", "yolov8m.pt")
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", 4))
CONF_THRESHOLD = 0.4

# COCO class name -> category
CATEGORIES = {
    "person": "persona",
    **{n: "veicolo" for n in ("bicycle", "car", "motorcycle", "bus", "truck")},
    **{n: "animale" for n in ("bird", "cat", "dog", "horse", "sheep", "cow",
                              "elephant", "bear", "zebra", "giraffe")},
}
