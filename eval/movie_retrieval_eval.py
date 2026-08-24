"""
Movie-level retrieval eval: held-out post -> movie.

The product embeds a query (a post / mood) and retrieves MOVIES, so the movie
representation must be scored at the movie level — which the post->post lift
in eval_embeddings.py does not do. This is that missing measuring stick.

Honest protocol (no leakage):
  1. Split posts 80/20 by post (fixed seed).
  2. Build the movie index from TRAIN posts only, with a pluggable edge weight.
     A test post never contributed to any movie vector, so retrieving its
     movies is a real generalization test.
  3. For each TEST post, rank all movies by cosine to the post vector.
     Relevant = the post's tmdb_ids that exist in the train-built index.
     Exclude test posts with an empty relevant set (report the count).
  4. Report MRR, Hit@10, Recall@5, Recall@10, and (using the upvote join)
     an agreement-weighted Recall@10 that rewards retrieving the movies the
     crowd most strongly agreed fit the post.

Weight schemes (how a post's edges contribute to its movies' vectors):
  uniform      w = 1
  log_shotgun  w = 1 / log2(1 + n_movies_in_post)   [current production]
  upvote       w = within-post upvote agreement of that (post, movie) edge

Usage:
  python eval/movie_retrieval_eval.py                         # current prod scheme
  python eval/movie_retrieval_eval.py --scheme upvote --name upvote_weighted
  python eval/movie_retrieval_eval.py --compare               # all schemes, one table
"""

import argparse
import json
import logging
import math
import os
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "eval_results")
POSTS_FILE = os.path.join(DATA_DIR, "posts_with_hybrid_vectors.json")
EDGES_FILE = os.path.join(DATA_DIR, "movie_edge_scores.jsonl")

VAL_FRACTION = 0.2
SEED = 42
KS = (5, 10)


def has_signal(p):
    has_text = isinstance(p.get("descriptors"), list) and len(p["descriptors"]) > 0
    return has_text or bool(p.get("image_exists"))


def load_posts():
    posts = []
    with open(POSTS_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (isinstance(p.get("combined_vector"), list) and
                    isinstance(p.get("tmdb_ids"), list) and p["tmdb_ids"] and has_signal(p)):
                posts.append(p)
    logging.info(f"Loaded {len(posts)} posts (signal + >=1 movie).")
    return posts


def load_edge_agreement():
    """(post_id, tmdb_id) -> within-post agreement in [0,1], from the upvote join."""
    agree = {}
    if not os.path.exists(EDGES_FILE):
        logging.warning("No movie_edge_scores.jsonl — upvote scheme/metric unavailable.")
        return agree
    with open(EDGES_FILE, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            a = e.get("agreement_norm")
            if a is not None:
                agree[(e["post_id"], e["tmdb_id"])] = a
    logging.info(f"Loaded {len(agree)} scored (post, movie) edges.")
    return agree


def edge_weight(scheme, post, tid, agree):
    if scheme == "uniform":
        return 1.0
    if scheme == "log_shotgun":
        return 1.0 / math.log2(1 + len(post["tmdb_ids"]))
    if scheme == "upvote":
        # Agreement in [0,1]; floor so a movie the crowd ignored still counts
        # a little (else its vector would be built from nothing).
        return 0.05 + agree.get((post["post_id"], tid), 0.0)
    raise ValueError(scheme)


def build_movie_index(train_posts, scheme, agree):
    sums, weights = {}, defaultdict(float)
    for p in train_posts:
        v = np.asarray(p["combined_vector"], dtype=np.float32)
        n = np.linalg.norm(v)
        if n == 0:
            continue
        v = v / n
        for tid in set(p["tmdb_ids"]):
            w = edge_weight(scheme, p, tid, agree)
            sums[tid] = v * w if tid not in sums else sums[tid] + v * w
            weights[tid] += w
    ids, mat = [], []
    for tid, s in sums.items():
        vec = s / weights[tid]
        nn = np.linalg.norm(vec)
        if nn > 0:
            ids.append(tid)
            mat.append(vec / nn)
    return ids, np.stack(mat).astype(np.float32)


def evaluate(posts, scheme, agree):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(posts))
    n_val = int(len(posts) * VAL_FRACTION)
    test = [posts[i] for i in order[:n_val]]
    train = [posts[i] for i in order[n_val:]]

    ids, mat = build_movie_index(train, scheme, agree)
    train_ids = set(ids)
    # train support per movie, for the tail breakdown
    support = defaultdict(int)
    for p in train:
        for t in set(p["tmdb_ids"]):
            support[t] += 1

    TIERS = [("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10-24", 10, 24), ("25+", 25, 10**9)]

    def tier_of(t):
        s = support[t]
        for name, lo, hi in TIERS:
            if lo <= s <= hi:
                return name
        return "1"

    mrr, hit10, rec = [], [], {k: [] for k in KS}
    wrec10, rel_sizes = [], []
    n_excluded = 0
    # per-relevant-(post,movie) rank+tier, pooled for the tail breakdown
    tier_hits = {name: [] for name, _, _ in TIERS}  # 1.0/0.0 in-top-10
    tier_rr = {name: [] for name, _, _ in TIERS}     # reciprocal rank

    for p in test:
        relevant = [t for t in set(p["tmdb_ids"]) if t in train_ids]
        if not relevant:
            n_excluded += 1
            continue
        rel_sizes.append(len(relevant))
        q = np.asarray(p["combined_vector"], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = mat @ q
        order_m = np.argsort(-sims)
        rank_of = {tid: r + 1 for r, tid in enumerate(ids[j] for j in order_m)}

        ranks = sorted(rank_of[t] for t in relevant)
        mrr.append(1.0 / ranks[0])
        for k in KS:
            hits = sum(1 for r in ranks if r <= k)
            rec[k].append(hits / len(relevant))
        hit10.append(1.0 if ranks[0] <= 10 else 0.0)

        for t in relevant:
            tn = tier_of(t)
            tier_hits[tn].append(1.0 if rank_of[t] <= 10 else 0.0)
            tier_rr[tn].append(1.0 / rank_of[t])

        gains = {t: agree.get((p["post_id"], t), 0.0) for t in relevant}
        total_g = sum(gains.values())
        if total_g > 0:
            got = sum(g for t, g in gains.items() if rank_of[t] <= 10)
            wrec10.append(got / total_g)

    by_tier = {name: {"n_pairs": len(tier_hits[name]),
                      "recall@10": round(float(np.mean(tier_hits[name])), 4) if tier_hits[name] else None,
                      "mrr": round(float(np.mean(tier_rr[name])), 4) if tier_rr[name] else None}
               for name, _, _ in TIERS}

    return {
        "scheme": scheme,
        "n_train_posts": len(train),
        "n_test_posts_scored": len(mrr),
        "n_test_excluded_no_relevant_in_index": n_excluded,
        "n_movies_in_index": len(ids),
        "relevant_set_size_mean": round(float(np.mean(rel_sizes)), 1),
        "MRR": round(float(np.mean(mrr)), 4),
        "Hit@10": round(float(np.mean(hit10)), 4),
        "Recall@5": round(float(np.mean(rec[5])), 4),
        "Recall@10": round(float(np.mean(rec[10])), 4),
        "AgreementWeightedRecall@10": round(float(np.mean(wrec10)), 4) if wrec10 else None,
        "by_support_tier": by_tier,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="log_shotgun", choices=["uniform", "log_shotgun", "upvote"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--compare", action="store_true", help="Run all three schemes.")
    args = ap.parse_args()

    posts = load_posts()
    agree = load_edge_agreement()
    schemes = ["uniform", "log_shotgun", "upvote"] if args.compare else [args.scheme]
    results = [evaluate(posts, s, agree) for s in schemes]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = args.name or ("movie_retrieval_compare" if args.compare else f"movie_retrieval_{args.scheme}")
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump(results if args.compare else results[0], f, indent=2)

    print("\n=== MOVIE-LEVEL RETRIEVAL (held-out post -> movie) ===")
    hdr = f"{'scheme':<13}{'MRR':>8}{'Hit@10':>9}{'Rec@5':>8}{'Rec@10':>8}{'wRec@10':>9}"
    print(hdr)
    for r in results:
        print(f"{r['scheme']:<13}{r['MRR']:>8}{r['Hit@10']:>9}{r['Recall@5']:>8}"
              f"{r['Recall@10']:>8}{str(r['AgreementWeightedRecall@10']):>9}")
    r0 = results[0]
    print(f"\ntest posts scored: {r0['n_test_posts_scored']}  "
          f"(excluded, movie not in train index: {r0['n_test_excluded_no_relevant_in_index']})")
    print(f"movies in index: {r0['n_movies_in_index']}  |  mean relevant/post: {r0['relevant_set_size_mean']}")
    print(f"\nSaved to {os.path.join(RESULTS_DIR, name + '.json')}")


if __name__ == "__main__":
    main()
