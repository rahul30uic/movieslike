"""
Export the vibe-space atlas: a 2D UMAP projection of the curated probe posts
(the 500 images already shipped as thumbnails), so the site can render the
embedding space as an explorable map.

Output: frontend/public/engine/atlas.json
    {points: [{f (thumb file), x, y (0..1), c (caption fragment)}]}

Usage:
    python pipeline/export_atlas.py
"""

import json
import logging
import os

import numpy as np
import umap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_DIR = os.path.join(REPO_ROOT, "frontend", "public", "engine")


def main():
    meta = json.load(open(os.path.join(ENGINE_DIR, "probe_posts.json"), encoding="utf-8"))
    vecs = np.fromfile(os.path.join(ENGINE_DIR, "probe_posts.bin"), dtype=np.float16)
    vecs = vecs.astype(np.float32).reshape(len(meta["posts"]), meta["dim"])

    logging.info(f"UMAP on {vecs.shape[0]} posts...")
    xy = umap.UMAP(n_neighbors=15, min_dist=0.15, metric="cosine",
                   random_state=42).fit_transform(vecs)
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))

    points = [
        {"f": p["f"], "x": round(float(x), 4), "y": round(float(y), 4), "c": p.get("c", "")}
        for p, (x, y) in zip(meta["posts"], xy)
    ]
    out = os.path.join(ENGINE_DIR, "atlas.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"points": points}, f)
    logging.info(f"Wrote {len(points)} atlas points to {out}.")


if __name__ == "__main__":
    main()
