"""
Encode post images with a DINO vision model (self-supervised features:
texture, lighting, composition — closer to "vibe" than CLIP-family semantics).

Row order matches post_modality_meta.json (same loader as encode_modalities.py)
so the vectors are directly interchangeable with the SigLIP image block.

Usage:
    python pipeline/encode_dino.py                                    # dinov2-base
    python pipeline/encode_dino.py --model facebook/dinov3-vitb16-pretrain-lvd1689m
"""

import argparse
import ast
import logging
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
INPUT_CSV = os.path.join(DATA_DIR, "final_dataset.csv")

DEFAULT_MODEL = "facebook/dinov2-base"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 64


def valid_image_paths(path_str):
    if not isinstance(path_str, str):
        return []
    paths = [os.path.join(DATA_DIR, p.strip()) for p in path_str.split("|") if p.strip()]
    return [p for p in paths if os.path.exists(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    df = pd.read_csv(INPUT_CSV)
    df["img_paths"] = df["image_local_path"].apply(valid_image_paths)

    logging.info(f"Loading {args.model} on {DEVICE}...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(DEVICE).eval()
    dim = model.config.hidden_size

    n = len(df)
    image_vecs = np.zeros((n, dim), dtype=np.float32)
    has_image = np.zeros(n, dtype=bool)

    flat_paths, img_rows = [], []
    for i, paths in enumerate(df["img_paths"]):
        for p in paths:
            flat_paths.append(p)
            img_rows.append(i)
    logging.info(f"Encoding {len(flat_paths)} images across {n} posts...")

    flat_vecs = np.zeros((len(flat_paths), dim), dtype=np.float32)
    ok = np.zeros(len(flat_paths), dtype=bool)
    with torch.no_grad():
        for start in tqdm(range(0, len(flat_paths), BATCH_SIZE), desc="DINO batches"):
            batch_paths = flat_paths[start:start + BATCH_SIZE]
            images, keep = [], []
            for j, p in enumerate(batch_paths):
                try:
                    images.append(Image.open(p).convert("RGB"))
                    keep.append(start + j)
                except Exception as e:
                    logging.warning(f"Skipping unreadable image {p}: {e}")
            if not images:
                continue
            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0]  # CLS token
            cls = torch.nn.functional.normalize(cls, dim=-1)
            flat_vecs[keep] = cls.cpu().numpy()
            ok[keep] = True

    for i in range(n):
        rows = [k for k, r in enumerate(img_rows) if r == i and ok[k]]
        if rows:
            avg = flat_vecs[rows].mean(axis=0)
            norm = np.linalg.norm(avg)
            if norm > 0:
                image_vecs[i] = avg / norm
                has_image[i] = True

    slug = args.model.split("/")[-1].replace("-", "_")
    out_path = os.path.join(DATA_DIR, f"post_{slug}_vectors.npz")
    np.savez_compressed(out_path, image_vecs=image_vecs, has_image=has_image)
    logging.info(f"Saved {has_image.sum()} image vectors ({dim}-dim) to {out_path}.")


if __name__ == "__main__":
    main()
