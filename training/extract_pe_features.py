"""Extract frozen PE Core-L features for every persona crop in an export manifest."""
import argparse
import csv
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image

from finetune import pad_square

ARCH = "vit_pe_core_large_patch14_336.fb"


@torch.inference_mode()
def extract(model, images, batch_size):
    cfg = timm.data.resolve_data_config({}, model=model)
    size = cfg["input_size"][1:]
    transform = T.Compose([
        T.Lambda(pad_square),
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(cfg["mean"], cfg["std"]),
    ])
    features = []
    for start in range(0, len(images), batch_size):
        x = torch.stack([transform(im) for im in images[start:start + batch_size]]).to(next(model.parameters()).device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            f = (model(x).float() + model(torch.flip(x, dims=[3])).float()) / 2
        features.append(f.cpu())
        print(f"processed {min(start + batch_size, len(images))}/{len(images)}", flush=True)
    f = torch.cat(features).numpy()
    norm = np.linalg.norm(f, axis=1, keepdims=True)
    if not np.isfinite(f).all() or np.any(norm == 0):
        raise ValueError("feature extraction produced invalid or zero vectors")
    return (f / norm).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="export directory; sibling <name>_manifest.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=12)
    a = ap.parse_args()
    data = Path(a.data)
    rows = [r for r in csv.DictReader(open(data.parent / f"{data.name}_manifest.csv"))
            if r["kind"] == "persona"]
    images = [Image.open(data / r["path"]).convert("RGB") for r in rows]
    model = timm.create_model(ARCH, pretrained=True, num_classes=0).eval().to(a.device)
    if a.device == "cuda":
        model = model.to(dtype=torch.bfloat16)
    feats = extract(model, images, a.batch_size)
    out = data.parent / f"feat_pe_core_L_{len(rows)}.npy"
    np.save(out, feats)
    print(f"saved {out}: shape={feats.shape}, dtype={feats.dtype}", flush=True)


if __name__ == "__main__":
    main()
