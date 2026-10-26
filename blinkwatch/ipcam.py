"""IP camera input: an Android phone running "Android IP Camera" (open source, MJPEG over HTTPS).

Keeps ONE connection to /video/mjpeg (the app answers 429 to many separate requests), looks for motion, lights the
phone torch while something moves, records a clip and registers it like a Blink clip: the watcher then classifies it.
Run:  python -m blinkwatch.ipcam   (needs IPCAM_URL / IPCAM_USER / IPCAM_PASS, see config.py)
"""
import base64
import collections
import http.client
import logging
import signal
import ssl
import sys
import time
import urllib.parse
from datetime import datetime, timezone

import cv2
import numpy as np

from . import config, store

log = logging.getLogger("ipcam")

ANALYSIS_FPS = 5  # motion is looked for on at most this many frames per second
PRE_ROLL_S = 2.0  # frames kept from before the motion started
POST_S = 8.0  # keep recording this long after the last motion
MAX_CLIP_S = 90.0
SETTLE_S = 2.0  # ignore motion after the torch changes: the whole scene changes brightness
MOTION_FRAMES = 2  # consecutive analysed frames with motion before a clip starts
MIN_TORCH_GAP_S = 20.0  # never toggle the torch faster than this (the app rate-limits requests)


class MotionDetector:
    """Frame differencing on a small blurred grayscale image against a slow running background.

    The threshold follows the measured noise (median of the difference), because at +10 exposure in the dark the sensor
    noise is large and a fixed threshold would fire all night.
    """

    def __init__(self, width=160, alpha=0.05, min_fraction=0.012, noise_k=6.0, min_threshold=12.0):
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
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        raise OSError("cannot open VideoWriter")
    for _, jpeg in frames:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        out.write(cv2.rotate(img, code) if code is not None else img)
    out.release()
    return seconds, len(frames)


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
    delay = 20.0
    while not stopping:
        try:
            cam.control(torch="off", exposure=config.IPCAM_EXPOSURE)
            session(cam, db, lambda: stopping)
            delay = 20.0
        except (OSError, ConnectionError, http.client.HTTPException) as e:
            wait = 150.0 if "429" in str(e) else delay  # the app answers 429 when it is asked too much
            log.warning("camera connection lost (%s); retry in %.0fs", e, wait)
            for _ in range(int(wait)):
                if stopping:
                    break
                time.sleep(1)
            delay = min(300.0, delay * 2)
    cam.control(torch="off")
    return 0


def session(cam, db, stop):
    detector, rec = MotionDetector(), Recorder()
    torch_on, torch_changed = False, time.monotonic() - 999
    motion_run, last_motion, last_analysis = 0, 0.0, 0.0
    frames_seen, t_first = 0, None
    log.info("connected to %s:%s", cam.host, cam.port)
    for jpeg in read_mjpeg(cam.connection, config.IPCAM_USER, config.IPCAM_PASS, stop):
        now = time.monotonic()
        frames_seen += 1
        t_first = t_first or now
        if frames_seen == 30:
            rec.fps = 30 / max(0.1, now - t_first)
            log.info("stream rate ~%.1f fps, %.0f KB/frame", rec.fps, len(jpeg) / 1024)
        rec.push(now, jpeg)
        if now - last_analysis < 1.0 / ANALYSIS_FPS:
            continue
        last_analysis = now
        gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        settling = now - torch_changed < SETTLE_S
        moving = detector.update(gray) and not settling
        if moving:
            last_motion = now
            motion_run += 1
        else:
            motion_run = 0
        if rec.clip is None and motion_run >= MOTION_FRAMES:
            rec.start(now)
            log.info("motion (%.1f%% changed, threshold %.0f): start clip", detector.last_fraction * 100, detector.last_threshold)
            if config.IPCAM_TORCH and not torch_on and now - cam.last_control > MIN_TORCH_GAP_S:
                torch_on = cam.control(torch="on")
                torch_changed = time.monotonic()
                detector.reset()
        elif rec.clip is not None and (now - last_motion > POST_S or now - rec.started > MAX_CLIP_S):
            frames = rec.finish()
            if torch_on:
                torch_on = not cam.control(torch="off")
                torch_changed = time.monotonic()
                detector.reset()
            save(db, frames)
    return


def save(db, frames):
    created = datetime.now(timezone.utc).astimezone()
    clip_id = f"ipcam_{created:%Y%m%d%H%M%S}"
    path = config.DOWNLOAD_PATH / f"{config.IPCAM_NAME}_{clip_id}_{created:%Y%m%d_%H%M%S}.mp4"
    try:
        seconds, count = write_clip(frames, path, config.IPCAM_ROTATE)
    except OSError:
        log.exception("could not write the clip")
        return
    store.add_clip(db, clip_id, config.IPCAM_NAME, created, path)
    log.info("saved %s (%.1fs, %d frames)", path.name, seconds, count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sys.exit(run())
