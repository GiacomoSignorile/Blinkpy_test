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
PERSON_NAMES = [n.strip() for n in os.environ.get("PERSON_NAMES", "Io,Mamma,Papà,Rossana,Domenico,Lucrezia").split(",")]
EMBED_MODEL = "vit_base_patch14_dinov2.lvd142m"  # benchmarked: best of the frozen-feature options
EMBED_DIM = 768  # embeddings of another size were made by an older model and get recomputed
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

# Notifications (ntfy)
NTFY_URL = os.environ.get("NTFY_URL")  # e.g. http://ntfy (inside docker); unset = notifications off
NTFY_TOPIC = os.environ.get("NTFY_TOPIC")
PUBLIC_URL = os.environ.get("PUBLIC_URL", "http://100.66.9.116:8765")  # how the phone reaches the web page
NOTIFY_CATEGORIES = [c.strip() for c in os.environ.get("NOTIFY_CATEGORIES", "persona,animale").split(",") if c.strip()]
NOTIFY_MUTE = {n.strip() for n in os.environ.get("NOTIFY_MUTE", "").split(",") if n.strip()}  # names that never notify
NOTIFY_COOLDOWN_ANIMAL = int(os.environ.get("NOTIFY_COOLDOWN_ANIMAL", 600))
# Below this confidence an identification is reported as "not recognised" instead of a name
# (measured on held-out clips: cats 50% -> ~93% right on 83% of cases; people 60% -> ~85% right on 33% of cases)
IDENTITY_MIN_CONF = {"animale": float(os.environ.get("IDENTITY_MIN_CONF_ANIMALE", 0.5)),
                     "persona": float(os.environ.get("IDENTITY_MIN_CONF_PERSONA", 0.6))}
NOTIFY_COOLDOWN = int(os.environ.get("NOTIFY_COOLDOWN", 120))  # seconds, per camera and category
NOTIFY_MAX_AGE = int(os.environ.get("NOTIFY_MAX_AGE", 600))  # skip clips older than this (backlog, reclassification)
NOTIFY_PRIORITY = {"persona": 4, "veicolo": 3, "animale": 2}
NOTIFY_TAGS = {"persona": "bust_in_silhouette", "veicolo": "car", "animale": "paw_prints"}
