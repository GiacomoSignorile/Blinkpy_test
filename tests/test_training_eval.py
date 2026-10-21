import sys
import unittest
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING))

from evaluate_ft_candidates import rolling_time_split  # noqa: E402


class RollingTimeSplitTest(unittest.TestCase):
    def test_validation_is_after_training_and_clips_do_not_cross(self):
        clips = ["a", "a", "b", "c", "d", "e", "f", "g", "h", "i"]
        times = ["01", "01", "02", "03", "04", "05", "06", "07", "08", "09"]
        train, validation = rolling_time_split(clips, times, cut=0.4, window=0.2)

        self.assertTrue(train)
        self.assertTrue(validation)
        train_clips = {clips[i] for i in train}
        validation_clips = {clips[i] for i in validation}
        self.assertFalse(train_clips & validation_clips)
        self.assertLess(max(times[i] for i in train), min(times[i] for i in validation))


if __name__ == "__main__":
    unittest.main()
