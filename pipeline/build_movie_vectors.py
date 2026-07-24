"""
Build movie-level vibe vectors by aggregating post vectors.

The unit of retrieval for the product is the MOVIE, not the post. Each movie's
vector is a weighted average of the vectors of the posts that recommend it:

  - Shotgun threads (a post recommending 50 movies) say little about any one
    movie, so a post's weight is 1 / log2(1 + n_movies_it_recommends).
  - Posts with neither descriptors nor a valid image have meaningless
    embeddings (encoded empty string) and are excluded — measured below
    random on the eval harness.

Outputs movie_vectors.json (JSONL): one record per movie with tmdb_id, title,
poster_path, vote_count, n_posts (support), and the unit-normalized vector.

Also prints a coverage audit (support distribution, popularity head) and a
qualitative sanity check: nearest-neighbor movies for a few anchors.

Usage:
    python build_movie_vectors.py
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
POSTS_FILE = os.path.join(DATA_DIR, "posts_with_vectors.json")
PHASE1_FILE = os.path.join(DATA_DIR, "phase1_extracted_movies_fresh_run.json")
EDGES_FILE = os.path.join(DATA_DIR, "movie_edge_scores.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "movie_vectors.json")

MIN_SUPPORT_FOR_DEMO = 3  # only show demo neighbors among movies with >= N posts
UPVOTE_FLOOR = 0.05       # a movie the crowd ignored still contributes a little
TRIM_FRAC = 0.25          # drop this fraction of most-off-vibe posts before averaging
TRIM_MIN_SUPPORT = 5      # only trim when there are enough posts to spare


def has_signal(post):
    """True if the post's embedding is based on real content (text or image)."""
    has_text = isinstance(post.get("descriptors"), list) and len(post["descriptors"]) > 0
    return has_text or bool(post.get("image_exists"))


def load_posts(posts_file):
    posts = []
    skipped_no_signal = 0
    with open(posts_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            vec = p.get("combined_vector")
            ids = p.get("tmdb_ids")
            if not (isinstance(vec, list) and isinstance(ids, list) and ids):
                continue
            if not has_signal(p):
                skipped_no_signal += 1
                continue
            posts.append(p)
    logging.info(
        f"Loaded {len(posts)} posts with signal; excluded {skipped_no_signal} "
        f"no-text-no-image posts (garbage embeddings)."
    )
    return posts


def load_movie_details():
    with open(PHASE1_FILE, "r", encoding="utf-8") as f:
        phase1 = json.load(f)
    details = {}
    for post in phase1:
        for rec in post.get("verified_recommendations", []) or []:
            if isinstance(rec, dict) and "tmdb_id" in rec:
                tid = rec["tmdb_id"]
                if tid not in details:
                    details[tid] = {
                        "title": rec.get("official_title", "Unknown Title"),
                        "poster_path": rec.get("poster_path"),
                        "vote_count": rec.get("vote_count"),
                    }
    logging.info(f"Loaded details for {len(details)} unique movies from phase 1.")
    return details


def load_edge_agreement():
    """(post_id, tmdb_id) -> within-post upvote agreement in [0,1]."""
    agree = {}
    if not os.path.exists(EDGES_FILE):
        return agree
    with open(EDGES_FILE, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            a = e.get("agreement_norm")
            if a is not None:
                agree[(e["post_id"], e["tmdb_id"])] = a
    return agree


def _weighted_mean(V, w):
    s = w.sum()
    if abs(s) < 1e-6:  # net-zero signed weights: fall back to plain mean
        return V.mean(0)
    return (V * w[:, None]).sum(0) / s


def _aggregate(V, w, aggregation):
    """Combine a movie's member post-vectors into one vector."""
    if aggregation == "trimmed" and len(V) >= TRIM_MIN_SUPPORT:
        c = _weighted_mean(V, w)
        c = c / (np.linalg.norm(c) + 1e-9)
        sims = V @ c
        keep = sims >= np.quantile(sims, TRIM_FRAC)  # drop the most off-vibe posts
        V, w = V[keep], w[keep]
    return _weighted_mean(V, w)


def build_vectors(posts, weighting="upvote", agree=None, aggregation="trimmed"):
    """Aggregate post vectors into per-movie vectors.

    weighting (per-edge contribution):
      upvote       edge weight = UPVOTE_FLOOR + within-post crowd agreement
                   (+8% MRR on held-out post->movie vs the legacy heuristic)
      log_shotgun  edge weight = 1 / log2(1 + n_movies_in_post) [legacy]

    aggregation (how a movie's posts combine):
      trimmed      weighted mean after dropping the TRIM_FRAC most off-vibe
                   posts (+13% MRR; robust to outlier recommendations)
      mean         plain weighted mean
    """
    agree = agree or {}
    members = defaultdict(list)   # tid -> [(unit_vec, weight), ...]

    for p in posts:
        vec = np.asarray(p["combined_vector"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        vec = vec / norm
        shotgun_w = 1.0 / math.log2(1 + len(p["tmdb_ids"]))
        for tid in set(p["tmdb_ids"]):
            w = (UPVOTE_FLOOR + agree.get((p["post_id"], tid), 0.0)
                 if weighting == "upvote" else shotgun_w)
            members[tid].append((vec, w))

    movies = {}
    for tid, mem in members.items():
        V = np.stack([m[0] for m in mem])
        w = np.asarray([m[1] for m in mem], dtype=np.float32)
        v = _aggregate(V, w, aggregation)
        n = np.linalg.norm(v)
        if n > 0:
            movies[tid] = {"vector": v / n, "n_posts": len(mem)}
    return movies


def coverage_audit(movies, details):
    supports = np.array([m["n_posts"] for m in movies.values()])
    print("\n=== COVERAGE AUDIT ===")
    print(f"Movies with a vibe vector : {len(movies)}")
    buckets = [(1, 1), (2, 4), (5, 9), (10, 24), (25, 10**9)]
    for lo, hi in buckets:
        n = int(np.sum((supports >= lo) & (supports <= hi)))
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
        print(f"  support {label:>6} posts : {n:5d} movies ({100 * n / len(movies):.1f}%)")

    print("\nTop 15 most-recommended movies (the popularity head):")
    top = sorted(movies.items(), key=lambda kv: -kv[1]["n_posts"])[:15]
    for tid, m in top:
        title = details.get(tid, {}).get("title", f"tmdb:{tid}")
        print(f"  {m['n_posts']:4d} posts  {title}")


def neighbor_demo(movies, details):
    ids = [tid for tid, m in movies.items() if m["n_posts"] >= MIN_SUPPORT_FOR_DEMO]
    mat = np.stack([movies[tid]["vector"] for tid in ids])
    by_title = {details.get(tid, {}).get("title", "").lower(): i for i, tid in enumerate(ids)}

    print(f"\n=== NEIGHBOR SANITY CHECK (among {len(ids)} movies with support >= {MIN_SUPPORT_FOR_DEMO}) ===")
    for query in ["blade runner", "paris, texas", "the thing", "lost in translation", "fargo"]:
        qi = by_title.get(query)
        if qi is None:
            match = next((t for t in by_title if query in t), None)
            if match is None:
                print(f"\n  '{query}': not in corpus with enough support, skipped")
                continue
            qi = by_title[match]
        sims = mat @ mat[qi]
        order = np.argsort(-sims)
        qtitle = details.get(ids[qi], {}).get("title", "?")
        print(f"\n  {qtitle}  →")
        shown = 0
        for j in order:
            if j == qi:
                continue
            t = details.get(ids[j], {}).get("title", f"tmdb:{ids[j]}")
            print(f"     {sims[j]:.3f}  {t}")
            shown += 1
            if shown == 5:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts-file", default=POSTS_FILE,
                    help="Post vectors JSONL (e.g. posts_with_hybrid_vectors.json).")
    ap.add_argument("--output", default=OUTPUT_FILE)
    ap.add_argument("--weighting", default="upvote", choices=["upvote", "log_shotgun"],
                    help="Edge weighting for aggregation (default: upvote, the eval winner).")
    ap.add_argument("--aggregation", default="trimmed", choices=["trimmed", "mean"],
                    help="How a movie's posts combine (default: trimmed, +13% MRR).")
    args = ap.parse_args()

    posts = load_posts(args.posts_file)
    details = load_movie_details()
    agree = load_edge_agreement() if args.weighting == "upvote" else {}
    if args.weighting == "upvote" and not agree:
        logging.warning("No movie_edge_scores.jsonl found — falling back to log_shotgun weighting.")
        args.weighting = "log_shotgun"
    logging.info(f"Aggregating with '{args.weighting}' weighting + '{args.aggregation}' "
                 f"({len(agree)} scored edges).")
    movies = build_vectors(posts, weighting=args.weighting, agree=agree,
                           aggregation=args.aggregation)

    with open(args.output, "w", encoding="utf-8") as f:
        for tid, m in movies.items():
            rec = {
                "tmdb_id": tid,
                **details.get(tid, {"title": "Unknown Title", "poster_path": None, "vote_count": None}),
                "n_posts": m["n_posts"],
                "vector": [round(float(x), 6) for x in m["vector"]],
            }
            f.write(json.dumps(rec) + "\n")
    logging.info(f"Wrote {len(movies)} movie vectors to {args.output}.")

    coverage_audit(movies, details)
    neighbor_demo(movies, details)


if __name__ == "__main__":
    main()
