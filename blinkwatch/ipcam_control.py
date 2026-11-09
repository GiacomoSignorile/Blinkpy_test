"""Settings and status of the IP camera, shared between the web UI (writes the settings) and the ipcam service (reads
them, publishes the status) through two small JSON files in DATA_DIR."""
import json
import os
import threading
import uuid
from datetime import datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import config

CONTROL_FILE = config.DATA_DIR / "ipcam_control.json"
STATUS_FILE = config.DATA_DIR / "ipcam_status.json"
REQUEST_FILE = config.DATA_DIR / "ipcam_preview_request.json"  # "take a fresh preview now", written by the web UI
PREVIEW_FILE = config.DATA_DIR / "ipcam_preview.jpg"  # latest frame, shown on the /cameras page

SENSITIVITY = {"low": 0.08, "medium": 0.04, "high": 0.02}  # share of the image that must change to count as motion
MAX_WINDOWS = 8
MAX_PAUSE_MIN = 7 * 24 * 60
ROTATIONS = (0, 90, 180, 270)  # clockwise degrees applied to clips and preview

DEFAULTS = {
    "enabled": True,
    "paused_until": None,  # ISO instant (UTC) until which recording is paused
    "sensitivity": "medium",
    "post_s": 8,  # keep recording this long after the last motion
    "exposure_day": config.IPCAM_EXPOSURE,
    "exposure_night": config.IPCAM_EXPOSURE_NIGHT,
    "torch": config.IPCAM_TORCH,
    "preview": True,  # keep a recent still of the camera for the web UI
    "rotate": config.IPCAM_ROTATE if config.IPCAM_ROTATE in (0, 90, 180, 270) else 0,
    "schedule": {"enabled": False, "windows": [{"days": [0, 1, 2, 3, 4, 5, 6], "start": "22:00", "end": "07:00"}]},
    "alerts": {"offline_min": 10, "battery_pct": 20},
}


def _tz():
    try:
        return ZoneInfo(config.TIMEZONE)
    except Exception:  # unknown zone name: fall back to UTC rather than crash the service
        return timezone.utc


def _hhmm(value, default):
    try:
        h, m = str(value).split(":")
        h, m = int(h), int(m)
        if 0 <= h < 24 and 0 <= m < 60:
            return f"{h:02d}:{m:02d}"
    except (ValueError, TypeError):
        pass
    return default


def _int(value, lo, hi, default):
    try:
        return max(lo, min(hi, int(value)))
    except (ValueError, TypeError):
        return default


def _bool(value, default):
    return value if isinstance(value, bool) else default


def _iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def normalize(raw, base=None):
    """Merge `raw` over `base` (or the defaults), keeping only known keys with clamped, well-typed values."""
    base = base or DEFAULTS
    raw = raw if isinstance(raw, dict) else {}
    out = {
        "enabled": _bool(raw.get("enabled"), base["enabled"]),
        "paused_until": _iso(raw["paused_until"]) if "paused_until" in raw else base["paused_until"],
        "sensitivity": raw.get("sensitivity") if raw.get("sensitivity") in SENSITIVITY else base["sensitivity"],
        "post_s": _int(raw.get("post_s", base["post_s"]), 3, 60, base["post_s"]),
        "exposure_day": _int(raw.get("exposure_day", base["exposure_day"]), -10, 10, base["exposure_day"]),
        "exposure_night": _int(raw.get("exposure_night", base["exposure_night"]), -10, 10, base["exposure_night"]),
        "torch": _bool(raw.get("torch"), base["torch"]),
        "preview": _bool(raw.get("preview"), base["preview"]),
        "rotate": raw["rotate"] if raw.get("rotate") in ROTATIONS and not isinstance(raw.get("rotate"), bool) else base["rotate"],
    }
    sched_raw = raw.get("schedule") if isinstance(raw.get("schedule"), dict) else {}
    sched_base = base["schedule"]
    windows = sched_base["windows"]
    if isinstance(sched_raw.get("windows"), list):
        windows = []
        for w in sched_raw["windows"][:MAX_WINDOWS]:
            if not isinstance(w, dict):
                continue
            days = sorted({d for d in (w.get("days") or []) if isinstance(d, int) and not isinstance(d, bool) and 0 <= d <= 6})
            if days:
                windows.append({"days": days, "start": _hhmm(w.get("start"), "22:00"), "end": _hhmm(w.get("end"), "07:00")})
    out["schedule"] = {"enabled": _bool(sched_raw.get("enabled"), sched_base["enabled"]), "windows": windows}
    alerts_raw = raw.get("alerts") if isinstance(raw.get("alerts"), dict) else {}
    out["alerts"] = {"offline_min": _int(alerts_raw.get("offline_min", base["alerts"]["offline_min"]), 1, 1440, base["alerts"]["offline_min"]),
                     "battery_pct": _int(alerts_raw.get("battery_pct", base["alerts"]["battery_pct"]), 0, 90, base["alerts"]["battery_pct"])}
    return out


def load():
    try:
        return normalize(json.loads(CONTROL_FILE.read_text()))
    except (OSError, ValueError):
        return normalize({})


def save(ctrl):
    tmp = CONTROL_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(ctrl, indent=1))
    os.replace(tmp, CONTROL_FILE)  # atomic: the reader never sees half a file


def update(patch):
    """Apply a partial change to the saved settings and return the result."""
    ctrl = normalize(patch, base=load())
    save(ctrl)
    return ctrl


def pause(minutes, now=None):
    """Pause recording for `minutes` (0 / None resumes)."""
    now = now or datetime.now(timezone.utc)
    until = None
    if minutes:
        until = (now + timedelta(minutes=_int(minutes, 1, MAX_PAUSE_MIN, 60))).isoformat()
    return update({"paused_until": until})


def _in_window(w, local):
    start = dtime.fromisoformat(w["start"])
    end = dtime.fromisoformat(w["end"])
    t, today = local.time(), local.weekday()
    if start == end:  # an empty range is read as "all day"
        return today in w["days"]
    if start < end:
        return today in w["days"] and start <= t < end
    # crosses midnight: the day listed is the one the window starts on
    if t >= start:
        return today in w["days"]
    return (today - 1) % 7 in w["days"] and t < end


def is_active(ctrl, now=None):
    """(recording?, reason). reason is one of: on, disabled, paused, outside_schedule."""
    now = now or datetime.now(timezone.utc)
    if not ctrl["enabled"]:
        return False, "disabled"
    if ctrl["paused_until"] and now < datetime.fromisoformat(ctrl["paused_until"]):
        return False, "paused"
    sched = ctrl["schedule"]
    if sched["enabled"]:
        local = now.astimezone(_tz())
        if not any(_in_window(w, local) for w in sched["windows"]):
            return False, "outside_schedule"
    return True, "on"


class ControlWatcher:
    """Re-reads the settings file only when it changes."""

    def __init__(self):
        self._mtime, self._ctrl = None, load()

    def get(self):
        try:
            m = CONTROL_FILE.stat().st_mtime_ns
        except OSError:
            m = None
        if m != self._mtime:
            self._mtime, self._ctrl = m, load()
        return self._ctrl


class Status:
    """What the ipcam service knows right now; flushed to STATUS_FILE for the web UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self.data = {}
        try:
            self.data = json.loads(STATUS_FILE.read_text())
        except (OSError, ValueError):
            pass
        # the alert flags survive a restart of the service, so a reboot does not repeat a notification
        self.data.setdefault("alerts_sent", {})

    def set(self, **kw):
        with self._lock:
            self.data.update(kw)

    def get(self, key, default=None):
        with self._lock:
            return self.data.get(key, default)

    def flush(self):
        with self._lock:
            payload = dict(self.data, written_at=datetime.now(timezone.utc).isoformat())
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, STATUS_FILE)


def read_status():
    try:
        return json.loads(STATUS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def request_preview():
    """Ask the ipcam service for a fresh preview; returns the request id the UI can wait for."""
    rid = uuid.uuid4().hex[:8]
    tmp = REQUEST_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"id": rid, "at": datetime.now(timezone.utc).isoformat()}))
    os.replace(tmp, REQUEST_FILE)
    return rid


def pending_request():
    """Id of the last preview request (the caller compares it with the last one it answered), or None."""
    try:
        return json.loads(REQUEST_FILE.read_text()).get("id")
    except (OSError, ValueError, AttributeError):
        return None
