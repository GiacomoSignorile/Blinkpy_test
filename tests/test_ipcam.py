import numpy as np

from blinkwatch.ipcam import MotionDetector, Recorder


def _frame(rng, noise, box=None):
    f = np.full((480, 640), 20, np.float32) + rng.normal(0, noise, (480, 640))
    if box:
        y, x = box
        f[y:y + 150, x:x + 130] += 80
    return np.clip(f, 0, 255).astype(np.uint8)


def test_noise_alone_is_not_motion_even_when_noisy():
    rng = np.random.default_rng(1)
    for noise in (4, 15, 30):
        d = MotionDetector()
        assert sum(d.update(_frame(rng, noise)) for _ in range(100)) == 0


def test_a_moving_block_is_motion():
    rng = np.random.default_rng(2)
    d = MotionDetector()
    for _ in range(20):
        d.update(_frame(rng, 10))
    assert sum(d.update(_frame(rng, 10, (100 + i * 10, 200))) for i in range(10)) >= 8


def test_pre_roll_keeps_only_the_last_seconds_and_clip_starts_from_it():
    r = Recorder()
    for i in range(100):
        r.push(i * 0.1, b"x%d" % i)  # 10 s of frames
    r.start(10.0)
    assert 15 <= len(r.clip) <= 25  # about PRE_ROLL_S (2 s) of frames at 10 fps
    r.push(10.1, b"new")
    assert r.finish()[-1][1] == b"new" and r.clip is None


CFG = {"offline_min": 10, "battery_pct": 20}


def test_offline_alert_waits_then_fires_once_then_announces_recovery():
    from blinkwatch.ipcam import evaluate_alerts
    ev, sent = evaluate_alerts({}, False, 5 * 60, None, CFG)
    assert ev == [] and not sent.get("offline")  # not silent long enough
    ev, sent = evaluate_alerts(sent, False, 11 * 60, None, CFG)
    assert len(ev) == 1 and "offline" in ev[0][0] and sent["offline"]
    ev, sent = evaluate_alerts(sent, False, 90 * 60, None, CFG)
    assert ev == []  # no repeat
    ev, sent = evaluate_alerts(sent, True, 0, None, CFG)
    assert len(ev) == 1 and "online" in ev[0][0] and not sent["offline"]
    ev, _ = evaluate_alerts(sent, True, 0, None, CFG)
    assert ev == []  # a normal online poll says nothing


def test_back_online_without_an_earlier_alert_stays_quiet():
    from blinkwatch.ipcam import evaluate_alerts
    ev, _ = evaluate_alerts({}, True, 0, 80, CFG)
    assert ev == []


def test_low_battery_alert_once_with_hysteresis():
    from blinkwatch.ipcam import evaluate_alerts
    ev, sent = evaluate_alerts({}, True, 0, 18, CFG)
    assert len(ev) == 1 and "batteria" in ev[0][0].lower()
    ev, sent = evaluate_alerts(sent, True, 0, 15, CFG)
    assert ev == [] and sent["battery"]
    ev, sent = evaluate_alerts(sent, True, 0, 25, CFG)  # charging, but not yet 10 points above the threshold
    assert sent["battery"]
    ev, sent = evaluate_alerts(sent, True, 0, 35, CFG)
    assert not sent["battery"]
    ev, _ = evaluate_alerts(sent, True, 0, 19, CFG)  # drops again: warns again
    assert len(ev) == 1


def test_preview_is_written_atomically_and_rotated(tmp_path):
    import cv2
    from blinkwatch.ipcam import save_preview
    img = np.zeros((60, 100, 3), np.uint8)  # 100 wide, 60 high
    jpeg = cv2.imencode(".jpg", img)[1].tobytes()
    out = tmp_path / "p.jpg"
    save_preview(jpeg, out)
    assert out.read_bytes() == jpeg and not list(tmp_path.glob("*.tmp"))
    save_preview(jpeg, out, rotate=90)
    h, w = cv2.imdecode(np.frombuffer(out.read_bytes(), np.uint8), cv2.IMREAD_COLOR).shape[:2]
    assert (w, h) == (60, 100)  # rotated a quarter turn
