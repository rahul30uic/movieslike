"""
Bootstrap confidence intervals for the headline retrieval metrics.

Resamples query posts with replacement (B=2000) and recomputes mean
rare-overlap lift@10, giving a 95% CI for each embedding version on the
full-corpus population. Neighbor sets are fixed (computed once); the
bootstrap captures query-sampling noise, which is the dominant term.

Usage:
    python eval/bootstrap_ci.py
"""

import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_embeddings import HEAD_QUANTILE, K, overlap_metrics  # noqa: E402
from collections import Counter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")

B = 2000
SEED = 42
RANDOM_PAIRS = 200_000


def load(path):
    posts = []
    for line in open(path, encoding="utf-8"):
        p = json.loads(line)
        if isinstance(p.get("combined_vector"), list) and p.get("tmdb_ids"):
            posts.append(p)
    return posts


def per_query_stats(posts):
    vectors = np.array([p["combined_vector"] for p in posts], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors /= norms
    id_sets = [set(p["tmdb_ids"]) for p in posts]

    freq = Counter(m for s in id_sets for m in s)
    counts = np.array(sorted(freq.values()))
    cutoff = counts[int(len(counts) * HEAD_QUANTILE)]
    head = {m for m, c in freq.items() if c >= cutoff}

    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    topk = np.argpartition(sims, -K, axis=1)[:, -K:]

    rare = np.zeros(len(posts))
    for i in range(len(posts)):
        rare[i] = np.mean([overlap_metrics(id_sets[i], id_sets[j], head)[2] for j in topk[i]])

    rng = np.random.default_rng(SEED)
    ii = rng.integers(0, len(posts), RANDOM_PAIRS)
    jj = rng.integers(0, len(posts), RANDOM_PAIRS)
    keep = ii != jj
    base = np.mean([overlap_metrics(id_sets[a], id_sets[b], head)[2]
                    for a, b in zip(ii[keep], jj[keep])])
    return rare, base


def bootstrap_ci(rare, base):
    rng = np.random.default_rng(SEED)
    n = len(rare)
    lifts = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        lifts[b] = rare[idx].mean() / base
    return float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))


def main():
    variants = {
        "hybrid_frozen": os.path.join(DATA_DIR, "posts_with_hybrid_vectors_frozen.json"),
        "hybrid_lora": os.path.join(DATA_DIR, "posts_with_hybrid_vectors.json"),
    }
    out = {}
    for name, path in variants.items():
        if not os.path.exists(path):
            logging.warning(f"{name}: {path} missing, skipped")
            continue
        posts = load(path)
        rare, base = per_query_stats(posts)
        lo, hi = bootstrap_ci(rare, base)
        point = float(rare.mean() / base)
        out[name] = {"n": len(posts), "lift_rare": round(point, 2),
                     "ci95": [round(lo, 2), round(hi, 2)]}
        logging.info(f"{name}: {point:.2f} [{lo:.2f}, {hi:.2f}] (n={len(posts)})")

    with open(os.path.join(RESULTS_DIR, "bootstrap_ci.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
