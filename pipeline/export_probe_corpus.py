"""
Export a curated, downsized subset of corpus images for the web probe.

Selection: farthest-point sampling over post vectors — the subset maximally
covers the vibe space, which makes better probe pairs than random sampling
and keeps the payload small.

Outputs:
  frontend/public/probe_imgs/<post_id>.jpg   (512px, q72, ~40-60KB each)
  frontend/public/engine/probe_posts.bin     (fp16 vectors, row-aligned)
  frontend/public/engine/probe_posts.json    ({dim, posts: [{id, f}]})

Usage:
    python pipeline/export_probe_corpus.py --count 500
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
POSTS_FILE = os.path.join(DATA_DIR, "posts_with_hybrid_vectors.json")
DATASET_CSV = os.path.join(DATA_DIR, "final_dataset.csv")
IMG_OUT = os.path.join(REPO_ROOT, "frontend", "public", "probe_imgs")
ENGINE_OUT = os.path.join(REPO_ROOT, "frontend", "public", "engine")

MAX_DIM = 512
QUALITY = 72


def farthest_point_sample(vectors, k, seed=42):
    rng = np.random.default_rng(seed)
    n = len(vectors)
    chosen = [int(rng.integers(n))]
    min_dist = 1 - vectors @ vectors[chosen[0]]
    for _ in range(k - 1):
        nxt = int(np.argmax(min_dist))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, 1 - vectors @ vectors[nxt])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=500)
    args = ap.parse_args()

    paths = {}
    for _, row in pd.read_csv(DATASET_CSV).iterrows():
        p = (row.get("image_local_path") or "") if isinstance(row.get("image_local_path"), str) else ""
        first = p.split("|")[0].strip()
        if first:
            ap_ = os.path.join(DATA_DIR, first)
            if os.path.exists(ap_):
                paths[row["post_id"]] = ap_

    posts, vecs = [], []
    for line in open(POSTS_FILE, encoding="utf-8"):
        p = json.loads(line)
        if p["post_id"] in paths and p.get("image_exists") and p.get("tmdb_ids"):
            posts.append(p)
            vecs.append(p["combined_vector"])
    vecs = np.asarray(vecs, dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    logging.info(f"Candidate pool: {len(posts)} image posts.")

    idx = farthest_point_sample(vecs, min(args.count, len(posts)))

    os.makedirs(IMG_OUT, exist_ok=True)
    os.makedirs(ENGINE_OUT, exist_ok=True)
    kept, kept_vecs, total = [], [], 0
    for i in idx:
        p = posts[i]
        try:
            img = Image.open(paths[p["post_id"]]).convert("RGB")
        except Exception:
            continue
        img.thumbnail((MAX_DIM, MAX_DIM))
        fname = f"{p['post_id']}.jpg"
        out = os.path.join(IMG_OUT, fname)
        img.save(out, "JPEG", quality=QUALITY, optimize=True)
        total += os.path.getsize(out)
        # Short caption fragment: shown as the live "current reading" while
        # the diagnosis narrows.
        caption = (p.get("caption") or "").strip()
        fragment = caption[:110].rsplit(" ", 1)[0] + "…" if len(caption) > 110 else caption
        kept.append({"id": p["post_id"], "f": fname, "c": fragment})
        kept_vecs.append(vecs[i].astype(np.float16))

    np.stack(kept_vecs).tofile(os.path.join(ENGINE_OUT, "probe_posts.bin"))
    with open(os.path.join(ENGINE_OUT, "probe_posts.json"), "w", encoding="utf-8") as f:
        json.dump({"dim": int(vecs.shape[1]), "posts": kept}, f)

    logging.info(f"Exported {len(kept)} probe images ({total / 1e6:.1f}MB) "
                 f"+ vectors ({np.stack(kept_vecs).nbytes / 1e6:.1f}MB).")


if __name__ == "__main__":
    main()
