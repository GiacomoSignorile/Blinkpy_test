import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from blinkwatch.person_model import PersonModel


class TinyBackbone(nn.Module):
    num_features = 3

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x):
        return self.proj(x).mean(dim=(2, 3))


class PersonModelTest(unittest.TestCase):
    def test_legacy_checkpoint_defaults_to_224(self):
        backbone = TinyBackbone()
        net = nn.Sequential(backbone, nn.Linear(backbone.num_features, 2))
        checkpoint = {
            "classes": ["A", "B"],
            "arch": "test-backbone",
            "arch_kwargs": {},
            "mean": [0.5, 0.5, 0.5],
            "std": [0.25, 0.25, 0.25],
            "state": net.state_dict(),
        }
        with patch("blinkwatch.person_model.torch.load", return_value=checkpoint), \
                patch("blinkwatch.person_model.timm.create_model", return_value=TinyBackbone()):
            model = PersonModel(Path("legacy.pt"))
            tensor = model.transform(Image.new("RGB", (80, 120), color=(90, 110, 130)))

        self.assertEqual(model.input_size, (224, 224))
        self.assertEqual(tuple(torch.as_tensor(tensor).shape), (3, 224, 224))

    def test_checkpoint_input_size_is_used_for_preprocessing_and_prediction(self):
        backbone = TinyBackbone()
        net = nn.Sequential(backbone, nn.Linear(backbone.num_features, 2))
        checkpoint = {
            "classes": ["A", "B"],
            "arch": "test-backbone",
            "arch_kwargs": {"img_size": 336},
            "mean": [0.5, 0.5, 0.5],
            "std": [0.25, 0.25, 0.25],
            "input_size": [336, 336],
            "state": net.state_dict(),
            "meta": {"model": "pe_core_L"},
        }
        with patch("blinkwatch.person_model.torch.load", return_value=checkpoint), \
                patch("blinkwatch.person_model.timm.create_model", return_value=TinyBackbone()):
            model = PersonModel(Path("candidate.pt"))
            tensor = model.transform(Image.new("RGB", (80, 120), color=(90, 110, 130)))
            prediction = model.predict(np.zeros((120, 80, 3), dtype=np.uint8))

        self.assertEqual(model.input_size, (336, 336))
        self.assertEqual(tuple(torch.as_tensor(tensor).shape), (3, 336, 336))
        self.assertIn(prediction[0], ("A", "B"))
        self.assertTrue(np.isfinite(prediction[1:]).all())


if __name__ == "__main__":
    unittest.main()
