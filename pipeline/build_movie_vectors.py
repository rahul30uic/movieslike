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
OUTPUT_FILE = os.path.join(DATA_DIR, "movie_vectors.json")

MIN_SUPPORT_FOR_DEMO = 3  # only show demo neighbors among movies with >= N posts


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


def build_vectors(posts):
    sums = defaultdict(lambda: None)
    weights = defaultdict(float)
    support = defaultdict(int)

    for p in posts:
        vec = np.asarray(p["combined_vector"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        vec = vec / norm
        w = 1.0 / math.log2(1 + len(p["tmdb_ids"]))
        for tid in set(p["tmdb_ids"]):
            sums[tid] = vec * w if sums[tid] is None else sums[tid] + vec * w
            weights[tid] += w
            support[tid] += 1

    movies = {}
    for tid, s in sums.items():
        v = s / weights[tid]
        n = np.linalg.norm(v)
        if n > 0:
            movies[tid] = {"vector": v / n, "n_posts": support[tid]}
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
    args = ap.parse_args()

    posts = load_posts(args.posts_file)
    details = load_movie_details()
    movies = build_vectors(posts)

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
