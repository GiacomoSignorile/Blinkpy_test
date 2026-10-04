"""Turn a crop into a unit-length feature vector (DINOv2), used to suggest labels."""
import cv2
import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image

from . import config


def pad_square(img, fill=(124, 116, 104)):
    """Letterbox to a square so tall crops (people) are not squashed or cropped."""
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), fill)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


class Embedder:
    def __init__(self):
        self.model = timm.create_model(config.EMBED_MODEL, pretrained=True, num_classes=0, img_size=224).eval()
        self.transform = T.Compose([
            T.Lambda(pad_square), T.Resize((224, 224)), T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def embed(self, bgr) -> np.ndarray:
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            v = self.model(self.transform(img).unsqueeze(0))[0].numpy().astype(np.float32)
        return v / np.linalg.norm(v)
