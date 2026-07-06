"""
Encode every post's text and images with SigLIP, saving the two modality
vectors SEPARATELY (unlike generate_embeddings_and_clusters.py, which only
saves the 50/50 average). This lets us sweep fusion weights offline without
re-encoding.

Outputs:
  post_modality_vectors.npz : text_vecs, image_vecs (unit-norm; zero row when
                              the modality is missing), has_text, has_image
  post_modality_meta.json   : JSONL — post_id, descriptors, tmdb_ids (aligned
                              row-by-row with the npz arrays)

Usage:
    python encode_modalities.py
"""

import ast
import json
import logging
import os

import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
INPUT_CSV = os.path.join(DATA_DIR, "final_dataset.csv")
NPZ_OUT = os.path.join(DATA_DIR, "post_modality_vectors.npz")
META_OUT = os.path.join(DATA_DIR, "post_modality_meta.json")

MODEL_NAME = "google/siglip-base-patch16-224"
DEVICE = "mps"
BATCH_SIZE = 128


def safe_list(val):
    if pd.isna(val):
        return []
    try:
        v = ast.literal_eval(val)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


def valid_image_paths(path_str):
    if not isinstance(path_str, str):
        return []
    paths = [os.path.join(DATA_DIR, p.strip()) for p in path_str.split("|") if p.strip()]
    return [p for p in paths if os.path.exists(p)]


def main():
    df = pd.read_csv(INPUT_CSV)
    for col in ["descriptors", "tmdb_ids"]:
        df[col] = df[col].apply(safe_list)
    df["img_paths"] = df["image_local_path"].apply(valid_image_paths)
    logging.info(f"Loaded {len(df)} posts.")

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    dim = model.get_sentence_embedding_dimension() or 768

    n = len(df)
    text_vecs = np.zeros((n, dim), dtype=np.float32)
    image_vecs = np.zeros((n, dim), dtype=np.float32)
    has_text = np.zeros(n, dtype=bool)
    has_image = np.zeros(n, dtype=bool)

    # --- Text: encode all non-empty descriptor strings in batches ---
    texts, text_rows = [], []
    for i, tags in enumerate(df["descriptors"]):
        if tags:
            texts.append(" ".join(tags))
            text_rows.append(i)
    logging.info(f"Encoding text for {len(texts)} posts...")
    if texts:
        tv = model.encode(texts, batch_size=BATCH_SIZE, convert_to_numpy=True,
                          show_progress_bar=True)
        text_vecs[text_rows] = normalize(tv, axis=1)
        has_text[text_rows] = True

    # --- Images: encode flat list, then mean-pool per post ---
    flat_paths, img_rows = [], []
    for i, paths in enumerate(df["img_paths"]):
        for p in paths:
            flat_paths.append(p)
            img_rows.append(i)
    logging.info(f"Encoding {len(flat_paths)} images across {n} posts...")

    flat_vecs = np.zeros((len(flat_paths), dim), dtype=np.float32)
    ok = np.zeros(len(flat_paths), dtype=bool)
    for start in tqdm(range(0, len(flat_paths), BATCH_SIZE), desc="Image batches"):
        batch_paths = flat_paths[start:start + BATCH_SIZE]
        images, keep = [], []
        for j, p in enumerate(batch_paths):
            try:
                images.append(Image.open(p).convert("RGB"))
                keep.append(start + j)
            except Exception as e:
                logging.warning(f"Skipping unreadable image {p}: {e}")
        if images:
            iv = model.encode(images, batch_size=BATCH_SIZE, convert_to_numpy=True,
                              show_progress_bar=False)
            flat_vecs[keep] = normalize(iv, axis=1)
            ok[keep] = True

    for i in range(n):
        rows = [k for k, r in enumerate(img_rows) if r == i and ok[k]]
        if rows:
            avg = flat_vecs[rows].mean(axis=0)
            norm = np.linalg.norm(avg)
            if norm > 0:
                image_vecs[i] = avg / norm
                has_image[i] = True

    np.savez_compressed(NPZ_OUT, text_vecs=text_vecs, image_vecs=image_vecs,
                        has_text=has_text, has_image=has_image)
    with open(META_OUT, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({
                "post_id": row["post_id"],
                "descriptors": row["descriptors"],
                "tmdb_ids": row["tmdb_ids"],
            }) + "\n")

    logging.info(f"Saved {n} rows: text={has_text.sum()}, image={has_image.sum()}, "
                 f"both={(has_text & has_image).sum()}, "
                 f"neither={(~has_text & ~has_image).sum()}")
    logging.info(f"Wrote {NPZ_OUT} and {META_OUT}.")


if __name__ == "__main__":
    main()
