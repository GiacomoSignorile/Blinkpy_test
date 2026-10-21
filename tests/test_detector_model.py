"""The default detector and container-preloaded checkpoint must match."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from blinkwatch import config


class DetectorModelTest(unittest.TestCase):
    def test_default_detector_is_preloaded_yolo26_medium(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("YOLO_MODEL", None)
            from importlib import reload
            self.assertEqual(reload(config).MODEL, "yolo26m.pt")
        dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
        self.assertIn("YOLO('yolo26m.pt')", dockerfile)


if __name__ == "__main__":
    unittest.main()
