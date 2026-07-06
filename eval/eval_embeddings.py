"""
Embedding evaluation harness.

Measures how well the post embedding space captures "vibe" using the one
ground-truth signal we have: two posts that recommend overlapping movies
very likely share a vibe.

For each post, retrieve its top-k nearest neighbors by cosine similarity and
measure movie overlap with the query post, compared against a random-pairs
baseline. Reports:

  - any-overlap@k : fraction of neighbors sharing >= 1 movie with the query
  - jaccard@k     : mean Jaccard similarity of tmdb_id sets
  - rare-overlap@k: any-overlap counting only movies OUTSIDE the global
                    popularity head (the long tail is what the product
                    actually promises)
  - lift          : each metric divided by its random baseline

Also breaks results down by modality regime (text+image / image-only /
text-only) to expose modality-gap artifacts.

Usage:
    python eval_embeddings.py                          # eval posts_with_vectors.json
    python eval_embeddings.py --embeddings-file F --name my_experiment

Results are printed and appended to eval_results/<name>.json so runs stay
comparable across embedding versions.
"""

import argparse
import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
DEFAULT_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "posts_with_vectors.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "eval_results")

K = 10                    # neighbors per query
RANDOM_PAIRS = 200_000    # pairs sampled for the baseline
HEAD_QUANTILE = 0.95      # movies above this global-frequency quantile = "head"
SEED = 42


def load_posts(path):
    """Load JSONL posts, keeping those with a vector and >= 1 verified movie."""
    posts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            vec = p.get("combined_vector")
            ids = p.get("tmdb_ids")
            if isinstance(vec, list) and isinstance(ids, list) and len(ids) > 0:
                posts.append(p)
    logging.info(f"Loaded {len(posts)} eligible posts (vector + >=1 tmdb_id).")
    return posts


def modality(post):
    has_text = isinstance(post.get("descriptors"), list) and len(post["descriptors"]) > 0
    has_image = bool(post.get("image_exists"))
    if has_text and has_image:
        return "text+image"
    if has_image:
        return "image-only"
    if has_text:
        return "text-only"
    return "neither"


def overlap_metrics(ids_a, ids_b, head):
    """Returns (any_overlap, jaccard, rare_overlap) for two tmdb_id sets."""
    inter = ids_a & ids_b
    union = ids_a | ids_b
    any_ov = 1.0 if inter else 0.0
    jac = len(inter) / len(union) if union else 0.0
    rare_ov = 1.0 if (inter - head) else 0.0
    return any_ov, jac, rare_ov


def evaluate(posts, k=K):
    rng = np.random.default_rng(SEED)
    n = len(posts)

    vectors = np.array([p["combined_vector"] for p in posts], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    id_sets = [set(p["tmdb_ids"]) for p in posts]
    regimes = [modality(p) for p in posts]

    # Global popularity head: movies above the HEAD_QUANTILE frequency quantile.
    freq = Counter(m for s in id_sets for m in s)
    counts = np.array(sorted(freq.values()))
    cutoff = counts[int(len(counts) * HEAD_QUANTILE)]
    head = {m for m, c in freq.items() if c >= cutoff}
    logging.info(
        f"{len(freq)} unique movies; head = {len(head)} movies with >= {cutoff} mentions."
    )

    # --- kNN metrics ---
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    topk = np.argpartition(sims, -k, axis=1)[:, -k:]

    per_regime = {r: [] for r in set(regimes)}
    all_rows = []
    for i in range(n):
        row = [overlap_metrics(id_sets[i], id_sets[j], head) for j in topk[i]]
        m = np.mean(row, axis=0)  # (any, jaccard, rare)
        all_rows.append(m)
        per_regime[regimes[i]].append(m)
    knn = np.mean(all_rows, axis=0)

    # --- Random baseline ---
    ii = rng.integers(0, n, RANDOM_PAIRS)
    jj = rng.integers(0, n, RANDOM_PAIRS)
    keep = ii != jj
    base = np.mean(
        [overlap_metrics(id_sets[a], id_sets[b], head) for a, b in zip(ii[keep], jj[keep])],
        axis=0,
    )

    def block(m, b):
        return {
            "any_overlap": round(float(m[0]), 4),
            "jaccard": round(float(m[1]), 4),
            "rare_overlap": round(float(m[2]), 4),
            "lift_any": round(float(m[0] / b[0]), 2) if b[0] else None,
            "lift_jaccard": round(float(m[1] / b[1]), 2) if b[1] else None,
            "lift_rare": round(float(m[2] / b[2]), 2) if b[2] else None,
        }

    results = {
        "k": k,
        "n_posts": n,
        "n_unique_movies": len(freq),
        "head_size": len(head),
        "random_baseline": {
            "any_overlap": round(float(base[0]), 4),
            "jaccard": round(float(base[1]), 4),
            "rare_overlap": round(float(base[2]), 4),
        },
        "knn": block(knn, base),
        "by_modality": {
            r: {"n": len(v), **block(np.mean(v, axis=0), base)}
            for r, v in sorted(per_regime.items())
            if v
        },
    }
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings-file", default=DEFAULT_EMBEDDINGS_FILE)
    ap.add_argument("--name", default="baseline_siglip_avg",
                    help="Label for this embedding version in eval_results/.")
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()

    posts = load_posts(args.embeddings_file)
    results = evaluate(posts, k=args.k)
    results["name"] = args.name
    results["embeddings_file"] = os.path.basename(args.embeddings_file)
    results["timestamp"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{args.name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
