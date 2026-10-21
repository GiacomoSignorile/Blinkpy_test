import unittest
from unittest.mock import patch

import cv2
import numpy as np

import blinkwatch.classifier as classifier_module
from blinkwatch.classifier import ClipClassifier


class FakeVideoCapture:
    def __init__(self, _path):
        self.position_ms = 0.0
        self.stuck = False
        self.recovered = False
        self.read_calls = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 10.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 200.0
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.position_ms
        return 0.0

    def set(self, prop, target_ms):
        if prop != cv2.CAP_PROP_POS_MSEC:
            return False
        # The demuxer reports success but snaps to the prior keyframe until a
        # later seek crosses the damaged GOP, matching the failing Blink clip.
        if target_ms < 6000:
            self.position_ms = 3000.0
            self.recovered = False
        else:
            self.position_ms = 6000.0
            self.recovered = True
        return True

    def read(self):
        self.read_calls += 1
        if self.read_calls > 500:
            raise AssertionError("classifier looped instead of advancing beyond the damaged GOP")
        if not self.stuck:
            if self.position_ms >= 4000:
                self.stuck = True
                return False, None
            self.position_ms += 100.0
            return True, np.zeros((16, 16, 3), dtype=np.uint8)
        if not self.recovered:
            if self.position_ms >= 4000:
                return False, None
            self.position_ms += 100.0
            return True, np.zeros((16, 16, 3), dtype=np.uint8)
        if self.position_ms >= 20000:
            return False, None
        self.position_ms += 100.0
        return True, np.full((16, 16, 3), 1, dtype=np.uint8)

    def release(self):
        pass


class FakeDetector:
    def __init__(self):
        self.saw_recovered_frame = False

    def __call__(self, frame, **_kwargs):
        self.saw_recovered_frame |= bool(frame[0, 0, 0])
        return [type("Result", (), {"boxes": [], "names": {}})()]


class ClassifierDecodeRecoveryTests(unittest.TestCase):
    def test_seek_rewound_to_bad_gop_still_advances_to_later_keyframe(self):
        capture = FakeVideoCapture("synthetic.mp4")
        detector = FakeDetector()
        classifier = ClipClassifier.__new__(ClipClassifier)
        classifier.model = detector

        with patch.object(classifier_module.cv2, "VideoCapture", return_value=capture):
            classifier.classify("synthetic.mp4")

        self.assertTrue(capture.recovered)
        self.assertTrue(detector.saw_recovered_frame)
        self.assertLess(capture.read_calls, 500)


if __name__ == "__main__":
    unittest.main()
