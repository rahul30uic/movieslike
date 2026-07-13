"""
Export compact deployment artifacts for the API.

Converts data/movie_vectors_hybrid.json (~260MB of JSON floats) into:
  deploy/data/movie_vectors_deploy.npz  — float16 matrix (~50MB)
  deploy/data/movies_meta.json          — JSONL metadata, vector-free

Only movies with min support are exported (they're the only recommendable
ones anyway), which further shrinks the artifact.

Usage:
    python pipeline/export_deploy_artifacts.py
"""

import json
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(REPO_ROOT, "data", "movie_vectors_hybrid.json")
OUT_DIR = os.path.join(REPO_ROOT, "deploy", "data")

MIN_SUPPORT = 2  # below this a movie can never be recommended (API floor is 3)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vecs, meta = [], []
    with open(SOURCE, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m["n_posts"] < MIN_SUPPORT:
                continue
            vecs.append(np.asarray(m.pop("vector"), dtype=np.float16))
            meta.append(m)

    matrix = np.stack(vecs)
    np.savez_compressed(os.path.join(OUT_DIR, "movie_vectors_deploy.npz"), vectors=matrix)
    with open(os.path.join(OUT_DIR, "movies_meta.json"), "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")

    size_mb = os.path.getsize(os.path.join(OUT_DIR, "movie_vectors_deploy.npz")) / 1e6
    logging.info(f"Exported {len(meta)} movies ({matrix.shape[1]}-dim, fp16) — {size_mb:.0f}MB npz.")


if __name__ == "__main__":
    main()
