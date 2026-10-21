"""Who is this person? CLIP-L fine-tuned on the human labels (trained by training/train_final.py).

Benchmark on days the model had not seen: ~78% balanced accuracy (frozen DINOv2 + logistic regression: ~52-67%).
The checkpoint lives outside the image, in data/models/, so retraining does not need a rebuild.
"""
import logging
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from . import config
from .embedder import pad_square

log = logging.getLogger("person_model")


class PersonModel:
    def __init__(self, path: Path):
        ck = torch.load(path, map_location="cpu", weights_only=True)  # plain tensors and lists only
        self.classes = ck["classes"]
        self.meta = ck.get("meta", {})
        self.name = path.stem  # e.g. persona_clipL_v1, saved next to every prediction
        backbone = timm.create_model(ck["arch"], pretrained=False, num_classes=0, **ck["arch_kwargs"])
        net = nn.Sequential(backbone, nn.Linear(backbone.num_features, len(self.classes)))
        net.load_state_dict({k: v.float() for k, v in ck["state"].items()})
        self.net = net.eval()
        self.transform = T.Compose([T.Lambda(pad_square), T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                                    T.ToTensor(), T.Normalize(tuple(ck["mean"]), tuple(ck["std"]))])
        torch.set_num_threads(config.TORCH_THREADS)

    @classmethod
    def load_if_present(cls):
        path = config.PERSON_MODEL
        if not path.exists():
            log.warning("no person model at %s: falling back to the small DINOv2 classifier", path)
            return None
        model = cls(path)
        log.info("person model %s loaded (%d classes, trained on %s crops)", model.name, len(model.classes),
                 model.meta.get("n_train"))
        return model

    @torch.no_grad()
    def predict(self, bgr):
        """(label, probability, margin over the second best) for one BGR crop."""
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        p = torch.softmax(self.net(self.transform(img).unsqueeze(0))[0], dim=0).numpy()
        order = np.argsort(p)[::-1]
        return self.classes[order[0]], float(p[order[0]]), float(p[order[0]] - p[order[1]])
