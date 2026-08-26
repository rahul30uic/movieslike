"""
Export per-movie affect vectors aligned to the two-tower item index, for the
browser's exclusion filter.

Row i of affect.bin corresponds to movies[i] in tt_movies.json (same order),
so the browser can look up a candidate's affect profile by its index.

Output: frontend/public/engine/affect.bin       (fp16, N x K)
        frontend/public/engine/affect_dims.json  ({dims})

Usage:
    python pipeline/export_affect.py
"""

import json
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(REPO, "frontend", "public", "engine")
AFFECT = os.path.join(REPO, "data", "movie_affect.json")


def main():
    movies = json.load(open(os.path.join(ENGINE, "tt_movies.json")))["movies"]
    affect = {int(k): v for k, v in json.load(open(AFFECT)).items()}
    dims = json.load(open(os.path.join(ENGINE, "affect_anchors.json")))["dims"]
    K = len(dims)

    mat = np.zeros((len(movies), K), dtype=np.float16)
    hit = 0
    for i, m in enumerate(movies):
        v = affect.get(m["id"])
        if v:
            mat[i] = v[:K]
            hit += 1
    mat.tofile(os.path.join(ENGINE, "affect.bin"))
    json.dump({"dims": dims}, open(os.path.join(ENGINE, "affect_dims.json"), "w"))
    logging.info(f"Wrote affect.bin {mat.shape} ({hit}/{len(movies)} with affect) "
                 f"and affect_dims.json ({K} dims).")


if __name__ == "__main__":
    main()
