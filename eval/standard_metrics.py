"""
Standard retrieval metrics for the production embedding space, computed on
IDENTICAL terms to the rare-movie-overlap lift in eval_embeddings.py.

This does not replace or modify that eval — it reuses its loading, its
neighbor/head logic, and its exact constants (k, rarity quantile, population,
random-pair seed) to report:

  1. Absolute rates behind the ~8.1x lift (random rate, neighbor rate, ratio).
     Taken directly from eval_embeddings.evaluate(), so the ratio reproduces
     the recorded lift by construction — a consistency check.
  2. Recall@5, Recall@10, MRR on the same relevance signal (a post is
     relevant to a query if they share >=1 rare movie, rare = outside the
     top-5% frequency head, exactly as the lift metric defines it).
  3. A sensitivity check: lift recomputed with "rare" redefined as support
     bands [5,10], [5,20], [5,50]. NOTE this is a DIFFERENT rarity notion
     than the production metric (absolute support range vs. quantile head
     exclusion); the production definition is included as a reference row.

Output: printed summary + eval_results/standard_metrics.json. Nothing in
eval_results/ is overwritten except this file.

Usage:
    python eval/standard_metrics.py
"""

import json
import os
from collections import Counter

import numpy as np

# Reuse the existing eval's loading, helpers, and exact constants — do not
# re-implement any of them here.
from eval_embeddings import (
    HEAD_QUANTILE,
    K,
    RANDOM_PAIRS,
    RESULTS_DIR,
    SEED,
    evaluate,
    load_posts,
    overlap_metrics,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROD_FILE = os.path.join(DATA_DIR, "posts_with_hybrid_vectors.json")
OUT_FILE = os.path.join(RESULTS_DIR, "standard_metrics.json")

SUPPORT_BANDS = [(5, 10), (5, 20), (5, 50)]


def build_head(id_sets):
    """The production 'head': movies at/above the HEAD_QUANTILE frequency
    cutoff. Reproduces eval_embeddings.evaluate()'s head construction exactly."""
    freq = Counter(m for s in id_sets for m in s)
    counts = np.array(sorted(freq.values()))
    cutoff = counts[int(len(counts) * HEAD_QUANTILE)]
    head = {m for m, c in freq.items() if c >= cutoff}
    return freq, head, int(cutoff)


def cosine_ranking(posts):
    """Full descending neighbor ranking per query (self excluded via -inf
    diagonal), using the same cosine construction as eval_embeddings."""
    vectors = np.array([p["combined_vector"] for p in posts], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    # Descending order; self lands last because its similarity is -inf.
    ranking = np.argsort(-sims, axis=1)
    return ranking


def retrieval_metrics(posts, id_sets, head, ranking):
    """Recall@5, Recall@10, MRR. Relevant(q) = other posts sharing >=1 rare
    movie (rare = outside head), the same signal the lift's rare_overlap uses."""
    n = len(posts)
    recall5, recall10, rr, rel_sizes = [], [], [], []
    n_excluded = 0
    for i in range(n):
        # Relevant set: any shared movie that is not in the head.
        relevant = {
            j for j in range(n)
            if j != i and (id_sets[i] & id_sets[j]) - head
        }
        if not relevant:
            n_excluded += 1
            continue
        rel_sizes.append(len(relevant))
        order = ranking[i]
        order = order[order != i]  # defensive; self already sorts out
        top10 = order[:10]
        top5 = order[:5]
        recall5.append(len(set(top5.tolist()) & relevant) / len(relevant))
        recall10.append(len(set(top10.tolist()) & relevant) / len(relevant))
        # MRR: reciprocal rank of the first relevant neighbor in full ranking.
        rank = next((r + 1 for r, j in enumerate(order.tolist()) if j in relevant), None)
        rr.append(1.0 / rank if rank else 0.0)

    return {
        "n_queries_scored": len(recall5),
        "n_queries_excluded_empty_relevant": n_excluded,
        # Relevant sets are large (a query is relevant to every post sharing any
        # of its rare movies), so Recall@k is bounded far below 1 by set size —
        # report the distribution so the recall figures are interpretable.
        "relevant_set_size_mean": round(float(np.mean(rel_sizes)), 1),
        "relevant_set_size_median": int(np.median(rel_sizes)),
        "recall_at_5": round(float(np.mean(recall5)), 4),
        "recall_at_10": round(float(np.mean(recall10)), 4),
        "mrr": round(float(np.mean(rr)), 4),
    }


def band_lift(posts, id_sets, freq, ranking, lo, hi, rng):
    """Lift with 'rare' = shared movie whose global support is in [lo, hi].
    kNN uses the same k=10 neighbors; random baseline uses the same
    RANDOM_PAIRS and SEED as the production eval."""
    n = len(posts)
    band = {m for m, c in freq.items() if lo <= c <= hi}

    def hit(a, b):
        return 1.0 if (id_sets[a] & id_sets[b]) & band else 0.0

    knn_hits = []
    for i in range(n):
        neigh = ranking[i][:K]
        knn_hits.append(np.mean([hit(i, j) for j in neigh]))
    knn_rate = float(np.mean(knn_hits))

    ii = rng.integers(0, n, RANDOM_PAIRS)
    jj = rng.integers(0, n, RANDOM_PAIRS)
    keep = ii != jj
    base_rate = float(np.mean([hit(a, b) for a, b in zip(ii[keep], jj[keep])]))

    return {
        "band": [lo, hi],
        "n_movies_in_band": len(band),
        "neighbor_rate": round(knn_rate, 4),
        "random_rate": round(base_rate, 4),
        "lift": round(knn_rate / base_rate, 2) if base_rate else None,
    }


def main():
    posts = load_posts(PROD_FILE)
    id_sets = [set(p["tmdb_ids"]) for p in posts]
    freq, head, cutoff = build_head(id_sets)

    # --- 1. Absolute rates behind the lift (straight from the existing eval) ---
    ev = evaluate(posts)
    abs_rates = {
        "random_pairs_rare_rate": ev["random_baseline"]["rare_overlap"],
        "neighbor_pairs_rare_rate": ev["knn"]["rare_overlap"],
        "lift": ev["knn"]["lift_rare"],
        "k": ev["k"],
        "n_posts": ev["n_posts"],
    }

    # --- 2. Standard retrieval metrics on the same relevance signal ---
    ranking = cosine_ranking(posts)
    retrieval = retrieval_metrics(posts, id_sets, head, ranking)

    # --- 3. Sensitivity: lift under support-band rarity definitions ---
    rng = np.random.default_rng(SEED)
    bands = [band_lift(posts, id_sets, freq, ranking, lo, hi, rng) for lo, hi in SUPPORT_BANDS]

    result = {
        "embeddings_file": os.path.basename(PROD_FILE),
        "space": "LoRA-tuned hybrid, w_cap=0.3 (production)",
        "params": {
            "k": K,
            "rarity": "production: movie outside top-5% frequency head "
                      f"(freq < {cutoff}); {len(head)} head movies of {len(freq)}",
            "population": f"{len(posts)} posts with vector + >=1 tmdb_id, no modality filter",
            "random_pairs": RANDOM_PAIRS,
            "seed": SEED,
        },
        "absolute_rates": abs_rates,
        "retrieval_metrics": retrieval,
        "sensitivity_support_bands": {
            "note": "DIFFERENT rarity notion than production (absolute support "
                    "range, not quantile head exclusion). Production reference below.",
            "production_reference_lift": abs_rates["lift"],
            "bands": bands,
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    a = abs_rates
    print("\n=== ABSOLUTE RATES BEHIND THE LIFT (k=%d, production rarity) ===" % K)
    print(f"  random pairs:   {a['random_pairs_rare_rate'] * 100:.2f}%")
    print(f"  neighbor pairs: {a['neighbor_pairs_rare_rate'] * 100:.2f}%")
    print(f"  lift:           {a['lift']}x   (consistency check vs recorded 8.1x)")

    r = retrieval
    print("\n=== STANDARD RETRIEVAL METRICS (relevant = shares a rare movie) ===")
    print(f"  queries scored:   {r['n_queries_scored']}")
    print(f"  excluded (no relevant): {r['n_queries_excluded_empty_relevant']}")
    print(f"  relevant set size: mean {r['relevant_set_size_mean']}, median {r['relevant_set_size_median']}")
    print(f"  Recall@5:  {r['recall_at_5']}   (bounded low: ~5/{r['relevant_set_size_median']} median relevant)")
    print(f"  Recall@10: {r['recall_at_10']}")
    print(f"  MRR:       {r['mrr']}   (first relevant neighbor ~rank {1 / r['mrr']:.1f})")

    print("\n=== SENSITIVITY: LIFT UNDER SUPPORT-BAND RARITY ===")
    print(f"  production (quantile head) reference: {a['lift']}x")
    for b in bands:
        print(f"  support {b['band'][0]}-{b['band'][1]:>2} "
              f"({b['n_movies_in_band']} movies): "
              f"random {b['random_rate'] * 100:.2f}%, "
              f"neighbor {b['neighbor_rate'] * 100:.2f}%, lift {b['lift']}x")

    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
