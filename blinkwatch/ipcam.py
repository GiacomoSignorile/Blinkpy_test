"""IP camera input: an Android phone running "Android IP Camera" (open source, MJPEG over HTTPS).

Keeps ONE connection to /video/mjpeg (the app answers 429 to many separate requests), looks for motion, lights the
phone torch while something moves, records a clip and registers it like a Blink clip: the watcher then classifies it.
Run:  python -m blinkwatch.ipcam   (needs IPCAM_URL / IPCAM_USER / IPCAM_PASS, see config.py)
"""
import base64
import collections
import http.client
import json
import logging
import os
import signal
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone

import cv2
import numpy as np

from . import config, ipcam_control, notify, store

log = logging.getLogger("ipcam")
STATUS = ipcam_control.Status()

ANALYSIS_FPS = 5  # motion is looked for on at most this many frames per second
PRE_ROLL_S = 2.0  # frames kept from before the motion started
MAX_CLIP_S = 90.0
SETTLE_S = 2.0  # ignore motion after the torch changes: the whole scene changes brightness
MOTION_FRAMES = 2  # consecutive analysed frames with motion before a clip starts
DARK_BELOW = 45.0  # mean gray level (0-255) at the day exposure under which the scene counts as dark
BRIGHT_ABOVE = 140.0  # mean gray level at the night exposure above which it is day again (well above DARK_BELOW: hysteresis)
EXPOSURE_HOLD_S = 60.0  # the brightness must stay past the limit this long before the exposure changes
MIN_EXPOSURE_GAP_S = 300.0
PREVIEW_EVERY_S = 3.0  # how often the still shown in the web UI is refreshed
INFO_EVERY_S = 60.0  # how often the phone is asked for battery and Wi-Fi
MIN_TORCH_GAP_S = 20.0  # never toggle the torch faster than this (the app rate-limits requests)


class MotionDetector:
    """Frame differencing on a small blurred grayscale image against a slow running background.

    The threshold follows the measured noise (median of the difference), because at +10 exposure in the dark the sensor
    noise is large and a fixed threshold would fire all night.
    """

    def __init__(self, width=160, alpha=0.05, min_fraction=0.04, noise_k=6.0, min_threshold=12.0):
        self.width, self.alpha, self.min_fraction, self.noise_k, self.min_threshold = width, alpha, min_fraction, noise_k, min_threshold
        self.background = None
        self.last_fraction = 0.0
        self.last_threshold = min_threshold

    def reset(self):
        self.background = None

    def _small(self, gray):
        h, w = gray.shape
        small = cv2.resize(gray, (self.width, max(1, int(h * self.width / w))), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (7, 7), 0).astype(np.float32)

    def update(self, gray) -> bool:
        """Feed one grayscale frame; True when a meaningful part of the image changed."""
        small = self._small(gray)
        if self.background is None:
            self.background = small
            return False
        diff = np.abs(small - self.background)
        # 1.4826 * MAD ~ sigma of the noise; the real motion is a small area, so the median stays on the noise
        self.last_threshold = max(self.min_threshold, self.noise_k * 1.4826 * float(np.median(diff)))
        self.last_fraction = float((diff > self.last_threshold).mean())
        moving = self.last_fraction >= self.min_fraction
        if not moving:  # do not learn a moving object into the background
            cv2.accumulateWeighted(small, self.background, self.alpha)
        return moving


def read_mjpeg(conn_factory, user, password, stop):
    """Yield (jpeg_bytes) frames from /video/mjpeg by scanning for JPEG start/end markers."""
    conn = conn_factory()
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    conn.request("GET", "/video/mjpeg", headers={"Authorization": f"Basic {auth}", "Connection": "keep-alive"})
    resp = conn.getresponse()
    if resp.status != 200:
        conn.close()
        raise ConnectionError(f"HTTP {resp.status}")
    buf = b""
    try:
        while not stop():
            chunk = resp.read1(65536)
            if not chunk:
                raise ConnectionError("stream closed")
            buf += chunk
            while True:
                start = buf.find(b"\xff\xd8")
                if start < 0:
                    buf = buf[-1:]
                    break
                end = buf.find(b"\xff\xd9", start + 2)
                if end < 0:
                    buf = buf[start:]
                    break
                yield buf[start:end + 2]
                buf = buf[end + 2:]
    finally:
        conn.close()


class Camera:
    def __init__(self):
        u = urllib.parse.urlparse(config.IPCAM_URL)
        self.host, self.port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
        self.https = u.scheme == "https"
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False  # the app uses a self-signed certificate; this is a LAN-only phone
        self.ctx.verify_mode = ssl.CERT_NONE
        self.auth = base64.b64encode(f"{config.IPCAM_USER}:{config.IPCAM_PASS}".encode()).decode()
        self.last_control = 0.0

    def connection(self):
        if self.https:
            return http.client.HTTPSConnection(self.host, self.port, timeout=20, context=self.ctx)
        return http.client.HTTPConnection(self.host, self.port, timeout=20)

    def info(self):
        """The app's /info.json (battery, Wi-Fi, active camera), or None when the phone does not answer."""
        conn = self.connection()
        try:
            conn.request("GET", "/info.json", headers={"Authorization": f"Basic {self.auth}"})
            resp = conn.getresponse()
            return json.loads(resp.read()) if resp.status == 200 else None
        except (OSError, ValueError, http.client.HTTPException):
            return None
        finally:
            conn.close()

    def control(self, **params) -> bool:
        """Set an app control, e.g. torch=on, exposure=10. One short request each."""
        query = urllib.parse.urlencode({**params, "ts": int(time.time() * 1000)})
        conn = self.connection()
        try:
            conn.request("GET", "/?" + query, headers={"Authorization": f"Basic {self.auth}"})
            ok = conn.getresponse().status == 200
        except OSError as e:
            log.warning("control %s failed: %s", params, e)
            ok = False
        finally:
            conn.close()
        self.last_control = time.monotonic()
        return ok


class Recorder:
    """Pre-roll ring of JPEG frames plus the clip being written (frames are kept encoded until the clip is closed)."""

    def __init__(self, fps_guess=15.0):
        self.fps = fps_guess
        self.pre = collections.deque()
        self.clip = None  # list of (t, jpeg) while recording
        self.started = 0.0

    def push(self, t, jpeg):
        if self.clip is not None:
            self.clip.append((t, jpeg))
            return
        self.pre.append((t, jpeg))
        while self.pre and t - self.pre[0][0] > PRE_ROLL_S:
            self.pre.popleft()

    def start(self, t):
        self.clip = list(self.pre)
        self.pre.clear()
        self.started = t

    def finish(self):
        frames, self.clip = self.clip, None
        return frames


def write_clip(frames, path, rotate=0):
    """Encode the collected JPEG frames to an mp4 with the real frame rate; returns (seconds, count)."""
    first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotate)
    if code is not None:
        first = cv2.rotate(first, code)
    seconds = max(0.5, frames[-1][0] - frames[0][0])
    fps = max(1.0, min(30.0, len(frames) / seconds))
    h, w = first.shape[:2]
    raw = path.with_name(path.stem + ".raw.mp4")
    out = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        raise OSError("cannot open VideoWriter")
    for _, jpeg in frames:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        out.write(cv2.rotate(img, code) if code is not None else img)
    out.release()
    _to_h264(raw, path)
    return seconds, len(frames)


def _to_h264(raw, path):
    """OpenCV only writes MPEG-4 Part 2, which browsers and phone players do not play: re-encode to H.264.
    Falls back to the MPEG-4 file if ffmpeg is missing or fails."""
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                       check=True, timeout=120)
        raw.unlink(missing_ok=True)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("H.264 re-encode failed (%s): keeping MPEG-4 clip", e)
        raw.replace(path)


def save_preview(jpeg, path, rotate=0):
    """Write a frame as the web preview (atomically, so the UI never reads half a file); same rotation as the clips."""
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotate)
    if code is not None:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            ok, buf = cv2.imencode(".jpg", cv2.rotate(img, code), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                jpeg = buf.tobytes()
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(jpeg)
    os.replace(tmp, path)


def answerable_request():
    """Id of a preview request the web UI made and that has not been answered yet, else None."""
    req = ipcam_control.pending_request()
    return req if req and req != STATUS.get("preview_handled_id") else None


def grab_preview(cam, ctrl, req):
    """Answer a preview request while the camera is not recording: open the stream just long enough for one frame.
    The request is marked as handled even on failure, so a phone that is off is not hammered every two seconds."""
    try:
        for jpeg in read_mjpeg(cam.connection, config.IPCAM_USER, config.IPCAM_PASS, lambda: False):
            save_preview(jpeg, ipcam_control.PREVIEW_FILE, ctrl["rotate"])
            STATUS.set(preview_handled_id=req, preview_error=None)
            return
        STATUS.set(preview_handled_id=req, preview_error="nessun fotogramma ricevuto")
    except (OSError, ConnectionError, http.client.HTTPException) as e:
        log.warning("preview request failed: %s", e)
        STATUS.set(preview_handled_id=req, preview_error=str(e)[:100])


def evaluate_alerts(sent, online, silent_s, battery, cfg):
    """Which notifications are due. `sent` holds the flags of those already sent; returns (events, new_sent)."""
    sent = dict(sent)
    events = []
    if not online and silent_s >= cfg["offline_min"] * 60 and not sent.get("offline"):
        sent["offline"] = True
        events.append(("Telecamera IP offline", f"Il telefono non risponde da {int(silent_s // 60)} minuti: controlla Wi-Fi e batteria.", 4, ("warning",)))
    if online and sent.get("offline"):
        sent["offline"] = False
        events.append(("Telecamera IP di nuovo online", "Il telefono risponde di nuovo.", 3, ("white_check_mark",)))
    if battery is not None:
        if battery <= cfg["battery_pct"] and not sent.get("battery"):
            sent["battery"] = True
            events.append(("Telecamera IP: batteria bassa", f"Batteria al {battery}%: collega il caricabatterie.", 4, ("battery",)))
        elif battery >= cfg["battery_pct"] + 10:
            sent["battery"] = False
    return events, sent


class Monitor(threading.Thread):
    """Publishes the status file every few seconds and, once a minute, asks the phone for battery and Wi-Fi,
    mirrors the battery into the camera card of the /cameras page and sends the offline / low-battery alerts."""

    def __init__(self, cam, control, stop):
        super().__init__(daemon=True, name="ipcam-monitor")
        self.cam, self.control, self.stop = cam, control, stop

    def poll(self):
        info = self.cam.info()
        now = datetime.now(timezone.utc)
        if info:
            battery = info.get("batteryPercent")
            wifi = info.get("wifiStrength")
            STATUS.set(online=True, last_contact_at=now.isoformat(),
                       battery_pct=battery if isinstance(battery, int) and battery >= 0 else None,
                       wifi_level=wifi if isinstance(wifi, int) and wifi >= 0 else None,
                       camera_id=(info.get("settings") or {}).get("cameraId"))
            self.mirror_to_db(now)
        else:
            STATUS.set(online=False)
        silent = (now - datetime.fromisoformat(STATUS.get("last_contact_at") or STATUS.get("started_at"))).total_seconds()
        events, sent = evaluate_alerts(STATUS.get("alerts_sent", {}), bool(info), silent, STATUS.get("battery_pct"), self.control.get()["alerts"])
        STATUS.set(alerts_sent=sent)
        for title, message, priority, tags in events:
            try:
                notify.send_text(title, message, priority, tags)
            except Exception as e:  # a notification problem must never stop the camera
                log.warning("notification failed: %s", e)

    def mirror_to_db(self, now):
        """The generic camera card reads its battery from camera_status, like the Blink cameras."""
        try:
            db = store.connect()
            try:
                db.execute("INSERT INTO camera_status (camera, data, updated_at) VALUES (?,?,?) "
                           "ON CONFLICT(camera) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
                           (config.IPCAM_NAME, json.dumps({"battery_level": STATUS.get("battery_pct")}), now.isoformat()))
                db.commit()
            finally:
                db.close()
        except Exception as e:
            log.warning("could not mirror the status: %s", e)

    def run(self):
        last_poll = 0.0
        while not self.stop():
            if time.monotonic() - last_poll >= INFO_EVERY_S:
                last_poll = time.monotonic()
                try:
                    self.poll()
                except Exception:
                    log.exception("status poll failed")
            try:
                STATUS.flush()
            except OSError as e:
                log.warning("could not write the status file: %s", e)
            time.sleep(2)


def run():
    if not (config.IPCAM_URL and config.IPCAM_USER and config.IPCAM_PASS):
        log.error("IPCAM_URL / IPCAM_USER / IPCAM_PASS not set: nothing to do")
        return 1
    cam, db = Camera(), store.connect()
    config.DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    ctl = ipcam_control.ControlWatcher()
    STATUS.set(started_at=datetime.now(timezone.utc).isoformat(), streaming=False, recording=False)
    Monitor(cam, ctl, lambda: stopping).start()
    delay = 20.0
    while not stopping:
        ctrl = ctl.get()
        active, reason = ipcam_control.is_active(ctrl)
        STATUS.set(active=active, reason=reason)
        if not active:
            STATUS.set(streaming=False, recording=False)
            req = answerable_request()
            if req and ctrl["preview"]:
                grab_preview(cam, ctrl, req)
            time.sleep(2)
            continue
        try:
            cam.control(torch="off", exposure=ctrl["exposure_day"])
            session(cam, db, lambda: stopping, ctl)
            delay = 20.0
        except (OSError, ConnectionError, http.client.HTTPException) as e:
            STATUS.set(streaming=False, recording=False, last_error=str(e)[:120])
            wait = 150.0 if "429" in str(e) else delay  # the app answers 429 when it is asked too much
            log.warning("camera connection lost (%s); retry in %.0fs", e, wait)
            for _ in range(int(wait)):
                if stopping:
                    break
                time.sleep(1)
            delay = min(120.0, delay * 2)
    cam.control(torch="off")
    STATUS.set(streaming=False, recording=False)
    try:
        STATUS.flush()
    except OSError:
        pass
    return 0


def session(cam, db, stop, ctl):
    detector, rec = MotionDetector(), Recorder()
    ctrl = ctl.get()
    torch_on, torch_changed = False, time.monotonic() - 999
    motion_run, last_motion, last_analysis, last_preview = 0, 0.0, 0.0, -999.0
    last_request_check = 0.0
    frames_seen, t_first = 0, None
    night, past_limit_since, exposure_changed = False, None, time.monotonic()
    applied_exposure = ctrl["exposure_day"]
    log.info("connected to %s:%s", cam.host, cam.port)
    STATUS.set(streaming=True, last_error=None)
    for jpeg in read_mjpeg(cam.connection, config.IPCAM_USER, config.IPCAM_PASS, stop):
        now = time.monotonic()
        frames_seen += 1
        t_first = t_first or now
        if frames_seen == 30:
            rec.fps = 30 / max(0.1, now - t_first)
            log.info("stream rate ~%.1f fps, %.0f KB/frame", rec.fps, len(jpeg) / 1024)
            STATUS.set(fps=round(rec.fps, 1))
        rec.push(now, jpeg)
        if now - last_analysis < 1.0 / ANALYSIS_FPS:
            continue
        last_analysis = now
        ctrl = ctl.get()
        active, reason = ipcam_control.is_active(ctrl)
        STATUS.set(active=active, reason=reason, recording=rec.clip is not None, last_frame_at=datetime.now(timezone.utc).isoformat())
        if not active and rec.clip is None:  # switched off, paused or outside the schedule: let go of the stream
            log.info("recording off (%s): closing the stream", reason)
            break
        if ctrl["preview"]:
            asked = now - last_request_check >= 1.0 and answerable_request()
            last_request_check = now if now - last_request_check >= 1.0 else last_request_check
            if asked or now - last_preview >= PREVIEW_EVERY_S:
                last_preview = now
                try:
                    save_preview(jpeg, ipcam_control.PREVIEW_FILE, ctrl["rotate"])
                    if asked:
                        STATUS.set(preview_handled_id=asked, preview_error=None)
                except OSError as e:
                    log.warning("could not write the preview: %s", e)
                    if asked:
                        STATUS.set(preview_handled_id=asked, preview_error=str(e)[:100])
        elif ipcam_control.PREVIEW_FILE.exists():  # switched off: do not leave an old still of the house lying around
            ipcam_control.PREVIEW_FILE.unlink(missing_ok=True)
        detector.min_fraction = ipcam_control.SENSITIVITY[ctrl["sensitivity"]]
        wanted = ctrl["exposure_night"] if night else ctrl["exposure_day"]
        if wanted != applied_exposure and rec.clip is None and now - cam.last_control > MIN_TORCH_GAP_S:
            cam.control(exposure=wanted)
            applied_exposure = wanted
            exposure_changed = torch_changed = time.monotonic()
            detector.reset()
        gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        settling = now - torch_changed < SETTLE_S
        if rec.clip is None and now - exposure_changed > MIN_EXPOSURE_GAP_S:
            mean = float(gray.mean())
            if (mean < DARK_BELOW) if not night else (mean > BRIGHT_ABOVE):
                past_limit_since = past_limit_since or now
                if now - past_limit_since > EXPOSURE_HOLD_S:
                    night = not night
                    applied_exposure = ctrl["exposure_night"] if night else ctrl["exposure_day"]
                    cam.control(exposure=applied_exposure)
                    log.info("scene %s (mean gray %.0f): exposure -> %d", "dark" if night else "bright", mean, applied_exposure)
                    exposure_changed, past_limit_since = time.monotonic(), None
                    torch_changed = exposure_changed  # the whole scene changes brightness: ignore motion for a moment
                    detector.reset()
            else:
                past_limit_since = None
        moving = detector.update(gray) and not (settling or now - exposure_changed < SETTLE_S) and active
        if moving:
            last_motion = now
            motion_run += 1
            STATUS.set(last_motion_at=datetime.now(timezone.utc).isoformat())
        else:
            motion_run = 0
        if rec.clip is None and motion_run >= MOTION_FRAMES:
            rec.start(now)
            log.info("motion (%.1f%% changed, threshold %.0f): start clip", detector.last_fraction * 100, detector.last_threshold)
            if ctrl["torch"] and not torch_on and now - cam.last_control > MIN_TORCH_GAP_S:
                torch_on = cam.control(torch="on")
                torch_changed = time.monotonic()
                detector.reset()
        elif rec.clip is not None and (now - last_motion > ctrl["post_s"] or now - rec.started > MAX_CLIP_S):
            frames = rec.finish()
            if torch_on:
                torch_on = not cam.control(torch="off")
                torch_changed = time.monotonic()
                detector.reset()
            save(db, frames, ctrl["rotate"])
    STATUS.set(streaming=False, recording=False)
    return


def save(db, frames, rotate=0):
    created = datetime.now(timezone.utc).astimezone()
    clip_id = f"ipcam_{created:%Y%m%d%H%M%S}"
    path = config.DOWNLOAD_PATH / f"{config.IPCAM_NAME}_{clip_id}_{created:%Y%m%d_%H%M%S}.mp4"
    try:
        seconds, count = write_clip(frames, path, rotate)
    except OSError:
        log.exception("could not write the clip")
        return
    store.add_clip(db, clip_id, config.IPCAM_NAME, created, path)
    log.info("saved %s (%.1fs, %d frames)", path.name, seconds, count)
    STATUS.set(last_clip_at=created.astimezone(timezone.utc).isoformat(), last_clip=path.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sys.exit(run())
