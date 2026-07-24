"""
Aggregation bake-off: is the single weighted-mean centroid the right way to
represent a movie, or are we losing signal by averaging?

All variants use the shipped upvote edge weighting; only the AGGREGATION
differs. Scored on the same held-out post->movie retrieval as
movie_retrieval_eval.py (reused loaders/split/metrics).

  wmean       weighted mean, unit-normalized              [current production]
  trimmed     drop the 25% of posts furthest from the mean, re-average (robust)
  medoid      the actual member post closest to all others (robust, real obs.)
  multiproto  cluster a movie's posts into up to K vibe-modes (k-means); a
              movie is SEVERAL prototype vectors, scored by MAX cosine to the
              query (late interaction). Directly fixes multimodal movies.

Usage:
    python eval/aggregation_bakeoff.py
"""

import json
import os
import sys

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from movie_retrieval_eval import (  # noqa: E402
    KS, SEED, VAL_FRACTION, edge_weight, load_edge_agreement, load_posts,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")

# multi-prototype schedule: 1 prototype per ~10 supporting posts, capped.
POSTS_PER_PROTO = 10
MAX_PROTOS = 4
TRIM_FRAC = 0.25


def collect(train_posts, agree):
    """Per movie: stacked member unit-vectors and their upvote edge weights."""
    vecs, wts = {}, {}
    for p in train_posts:
        v = np.asarray(p["combined_vector"], dtype=np.float32)
        n = np.linalg.norm(v)
        if n == 0:
            continue
        v = v / n
        for tid in set(p["tmdb_ids"]):
            w = edge_weight("upvote", p, tid, agree)
            vecs.setdefault(tid, []).append(v)
            wts.setdefault(tid, []).append(w)
    return {tid: (np.stack(vecs[tid]), np.asarray(wts[tid], dtype=np.float32))
            for tid in vecs}


def wmean(V, w):
    v = (V * w[:, None]).sum(0) / w.sum()
    return [v]


def trimmed(V, w):
    if len(V) < 5:
        return wmean(V, w)
    c = (V * w[:, None]).sum(0)
    c /= np.linalg.norm(c) + 1e-9
    sims = V @ c
    keep = sims >= np.quantile(sims, TRIM_FRAC)
    return wmean(V[keep], w[keep])


def medoid(V, w):
    if len(V) < 3:
        return wmean(V, w)
    if len(V) > 200:  # cap pairwise cost
        idx = np.random.default_rng(0).choice(len(V), 200, replace=False)
        V, w = V[idx], w[idx]
    wpos = np.clip(w, 1e-6, None)  # centrality importance is non-negative
    sim = V @ V.T
    return [V[np.argmax(sim @ wpos)]]


def multiproto(V, w):
    k = min(MAX_PROTOS, max(1, len(V) // POSTS_PER_PROTO))
    if k <= 1 or len(V) < 2 * k:
        return wmean(V, w)
    wpos = np.clip(w, 1e-6, None)  # k-means sample weights must be non-negative
    km = KMeans(n_clusters=k, n_init=3, random_state=SEED).fit(V, sample_weight=wpos)
    protos = []
    for c in range(k):
        m = km.labels_ == c
        if m.any() and wpos[m].sum() > 0:
            protos.append((V[m] * wpos[m, None]).sum(0) / wpos[m].sum())
    return protos or wmean(V, w)


def randproto(V, w):
    """Control for multiproto: same K, but a RANDOM partition instead of
    k-means. If this matches multiproto, the gain is just the max-of-K
    scoring artifact, not real vibe-mode clustering."""
    k = min(MAX_PROTOS, max(1, len(V) // POSTS_PER_PROTO))
    if k <= 1 or len(V) < 2 * k:
        return wmean(V, w)
    wpos = np.clip(w, 1e-6, None)
    labels = np.random.default_rng(SEED).integers(0, k, len(V))
    protos = []
    for c in range(k):
        m = labels == c
        if m.any() and wpos[m].sum() > 0:
            protos.append((V[m] * wpos[m, None]).sum(0) / wpos[m].sum())
    return protos or wmean(V, w)


def build(method, movie_data):
    fn = {"wmean": wmean, "trimmed": trimmed, "medoid": medoid,
          "multiproto": multiproto, "randproto": randproto}[method]
    ids, rows, seg = [], [], []
    for mi, (tid, (V, w)) in enumerate(movie_data.items()):
        ids.append(tid)
        for pv in fn(V, w):
            n = np.linalg.norm(pv)
            if n > 0:
                rows.append(pv / n)
                seg.append(mi)
    return ids, np.stack(rows).astype(np.float32), np.asarray(seg)


def score_and_metrics(test, ids, protos, seg, agree):
    idx_movie = {t: i for i, t in enumerate(ids)}
    train_ids = set(ids)
    M = len(ids)
    mrr, hit10, rec = [], [], {k: [] for k in KS}
    wrec10 = []
    n_excl = 0
    for p in test:
        relevant = [t for t in set(p["tmdb_ids"]) if t in train_ids]
        if not relevant:
            n_excl += 1
            continue
        q = np.asarray(p["combined_vector"], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        s = protos @ q
        ms = np.full(M, -np.inf, dtype=np.float32)
        np.maximum.at(ms, seg, s)          # per-movie max over its prototypes
        order = np.argsort(-ms)
        rank_of = {ids[j]: r + 1 for r, j in enumerate(order)}
        ranks = sorted(rank_of[t] for t in relevant)
        mrr.append(1.0 / ranks[0])
        hit10.append(1.0 if ranks[0] <= 10 else 0.0)
        for k in KS:
            rec[k].append(sum(r <= k for r in ranks) / len(relevant))
        gains = {t: agree.get((p["post_id"], t), 0.0) for t in relevant}
        tg = sum(gains.values())
        if tg > 0:
            wrec10.append(sum(g for t, g in gains.items() if rank_of[t] <= 10) / tg)
    return {
        "MRR": round(float(np.mean(mrr)), 4),
        "Hit@10": round(float(np.mean(hit10)), 4),
        "Recall@5": round(float(np.mean(rec[5])), 4),
        "Recall@10": round(float(np.mean(rec[10])), 4),
        "wRecall@10": round(float(np.mean(wrec10)), 4),
        "n_prototypes": int(len(seg)),
        "n_movies": len(ids),
    }


def main():
    posts = load_posts()
    agree = load_edge_agreement()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(posts))
    n_val = int(len(posts) * VAL_FRACTION)
    test = [posts[i] for i in order[:n_val]]
    train = [posts[i] for i in order[n_val:]]
    movie_data = collect(train, agree)

    results = {}
    for method in ["wmean", "trimmed", "medoid", "randproto", "multiproto"]:
        ids, protos, seg = build(method, movie_data)
        results[method] = score_and_metrics(test, ids, protos, seg, agree)

    with open(os.path.join(RESULTS_DIR, "aggregation_bakeoff.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== AGGREGATION BAKE-OFF (held-out post -> movie, upvote weighting) ===")
    print(f"{'method':<12}{'MRR':>8}{'Hit@10':>9}{'Rec@10':>8}{'wRec@10':>9}{'protos':>9}")
    for m, r in results.items():
        tag = "  <- current" if m == "wmean" else ""
        print(f"{m:<12}{r['MRR']:>8}{r['Hit@10']:>9}{r['Recall@10']:>8}"
              f"{r['wRecall@10']:>9}{r['n_prototypes']:>9}{tag}")
    print(f"\nSaved to {os.path.join(RESULTS_DIR, 'aggregation_bakeoff.json')}")


if __name__ == "__main__":
    main()
