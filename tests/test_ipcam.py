import numpy as np

from blinkwatch.ipcam import MotionDetector, Recorder


def _frame(rng, noise, box=None):
    f = np.full((480, 640), 20, np.float32) + rng.normal(0, noise, (480, 640))
    if box:
        y, x = box
        f[y:y + 120, x:x + 100] += 80
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
