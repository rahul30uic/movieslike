"""
Export the movie index as static files for the in-browser engine:

  frontend/public/engine/vectors.bin  — row-major float16, one row per movie
  frontend/public/engine/movies.json  — {dim, movies: [{id,t,p,n,v}...]}
                                        (short keys keep the payload small)

Row i of vectors.bin corresponds to movies[i]. Vectors are unit-normalized
before quantization so the browser only does dot products.

Usage:
    python pipeline/export_web_index.py
"""

import json
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(REPO_ROOT, "data", "movie_vectors_hybrid.json")
OUT_DIR = os.path.join(REPO_ROOT, "frontend", "public", "engine")

MIN_SUPPORT = 2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vecs, meta = [], []
    with open(SOURCE, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m["n_posts"] < MIN_SUPPORT:
                continue
            v = np.asarray(m["vector"], dtype=np.float32)
            v /= np.linalg.norm(v) + 1e-9
            vecs.append(v.astype(np.float16))
            meta.append({
                "id": m["tmdb_id"],
                "t": m["title"],
                "p": m.get("poster_path"),
                "n": m["n_posts"],
                "v": m.get("vote_count") or 0,
            })

    matrix = np.stack(vecs)
    matrix.tofile(os.path.join(OUT_DIR, "vectors.bin"))
    with open(os.path.join(OUT_DIR, "movies.json"), "w", encoding="utf-8") as f:
        json.dump({"dim": int(matrix.shape[1]), "movies": meta}, f)

    logging.info(f"Exported {len(meta)} movies, dim {matrix.shape[1]}: "
                 f"vectors.bin {matrix.nbytes / 1e6:.0f}MB, "
                 f"movies.json {os.path.getsize(os.path.join(OUT_DIR, 'movies.json')) / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
