"""
Build the whitened low-dimensional "dial" space the probe reasons in.

The hybrid vibe space is 1536-D but its intrinsic dimensionality is tiny
(participation ratio ~5; ~11 in the caption half). A 5-question probe (~5
bits) can only localize a low-dim target, so we project into the top-K PCA
axes and WHITEN them (divide each by its singular value) — which also
rebalances the dominant SigLIP-anisotropy axis that otherwise carries 43% of
the variance.

Outputs:
  data/dial_space.npz              mean (D,), components (Kmax,D), scales (Kmax,)
  frontend/public/engine/dial_space.json   same, for in-browser projection

Project:     d = ((v - mean) @ components[:K].T) / scales[:K]
Reconstruct: v ~= mean + (d * scales[:K]) @ components[:K]

Usage:
    python pipeline/build_dial_space.py
"""

import json
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
POSTS_FILE = os.path.join(DATA_DIR, "posts_with_hybrid_vectors.json")
NPZ_OUT = os.path.join(DATA_DIR, "dial_space.npz")
JSON_OUT = os.path.join(REPO, "frontend", "public", "engine", "dial_space.json")

KMAX = 15  # save the top 15; consumers slice the first K they want


def main():
    V = []
    for line in open(POSTS_FILE, encoding="utf-8"):
        p = json.loads(line)
        if isinstance(p.get("combined_vector"), list):
            V.append(p["combined_vector"])
    V = np.asarray(V, dtype=np.float64)
    logging.info(f"{V.shape[0]} posts, {V.shape[1]} dims.")

    mean = V.mean(0)
    Vc = V - mean
    # SVD: rows of Wt are principal axes; singular values S set the scale.
    _, S, Wt = np.linalg.svd(Vc, full_matrices=False)
    n = V.shape[0]
    components = Wt[:KMAX]                      # (KMAX, D)
    scales = S[:KMAX] / np.sqrt(n - 1)          # per-dial std (whitening divisor)

    evr = (S ** 2) / (S ** 2).sum()
    logging.info(f"top-{KMAX} variance captured: {evr[:KMAX].sum()*100:.1f}% "
                 f"(dial-1 alone {evr[0]*100:.1f}%)")

    np.savez_compressed(NPZ_OUT, mean=mean.astype(np.float32),
                        components=components.astype(np.float32),
                        scales=scales.astype(np.float32))
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "dim": int(V.shape[1]),
            "k": KMAX,
            "mean": [round(float(x), 6) for x in mean],
            "components": [[round(float(x), 6) for x in row] for row in components],
            "scales": [round(float(x), 6) for x in scales],
        }, f)
    logging.info(f"Wrote {NPZ_OUT} and {JSON_OUT} "
                 f"({os.path.getsize(JSON_OUT)/1e6:.1f}MB json).")


if __name__ == "__main__":
    main()
