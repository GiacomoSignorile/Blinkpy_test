"""Turn a crop into a unit-length feature vector (DINOv2), used to suggest labels."""
import cv2
import numpy as np
import timm
import torch
from PIL import Image

from . import config


class Embedder:
    def __init__(self):
        self.model = timm.create_model(config.EMBED_MODEL, pretrained=True, num_classes=0, img_size=224).eval()
        cfg = timm.data.resolve_data_config({"input_size": (3, 224, 224)}, model=self.model)
        self.transform = timm.data.create_transform(**cfg)

    def embed(self, bgr) -> np.ndarray:
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            v = self.model(self.transform(img).unsqueeze(0))[0].numpy().astype(np.float32)
        return v / np.linalg.norm(v)
