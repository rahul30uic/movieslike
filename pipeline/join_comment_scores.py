"""
Join comment upvote scores (from fetch_comment_scores.py) onto the movie
mentions in final_dataset.csv, producing the upvote-enriched dataset.

Attribution is best-effort: for each verified movie in a post, we find the
comments in that post whose body mentions the title (normalized, word-boundary
match), and attach those comments' scores. Very short / ambiguous titles
("It", "Her", "Up") are flagged rather than substring-matched, since "her"
appears in ordinary text — those edges get match_confidence="ambiguous".

Outputs:
  data/final_dataset_with_upvotes.jsonl
      a copy of each final_dataset row + per-movie upvote attribution +
      post-level comment-score features. THE joined dataset.
  data/movie_edge_scores.jsonl
      one flat row per (post, movie) edge with score stats + within-post
      agreement, for modeling / weighting.
  data/movie_agreement.json
      per-movie aggregation across posts — "which movies are most agreed to
      fit their vibes" (consensus ranking).

Usage:
    python pipeline/join_comment_scores.py
"""

import ast
import csv
import json
import logging
import os
import re
import sys
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
COMMENTS_FILE = os.path.join(DATA_DIR, "comments_with_scores.jsonl")
DATASET_CSV = os.path.join(DATA_DIR, "final_dataset.csv")
REDDIT_CSV = os.path.join(DATA_DIR, "Reddit_data.csv")

OUT_DATASET = os.path.join(DATA_DIR, "final_dataset_with_upvotes.jsonl")
OUT_EDGES = os.path.join(DATA_DIR, "movie_edge_scores.jsonl")
OUT_AGREEMENT = os.path.join(DATA_DIR, "movie_agreement.json")

MIN_TITLE_LEN = 4  # normalized titles shorter than this are ambiguous to match


def normalize(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


_pattern_cache = {}


def title_pattern(norm_title):
    pat = _pattern_cache.get(norm_title)
    if pat is None:
        pat = re.compile(r"\b" + re.escape(norm_title) + r"\b")
        _pattern_cache[norm_title] = pat
    return pat


def safe_list(v):
    if not isinstance(v, str) or not v.strip():
        return []
    try:
        out = ast.literal_eval(v)
        return out if isinstance(out, list) else []
    except (ValueError, SyntaxError):
        return []


def load_comments_by_post():
    by_post = defaultdict(list)
    n = 0
    with open(COMMENTS_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            body = c.get("body") or ""
            if body in ("[deleted]", "[removed]", ""):
                continue
            by_post[c["post_id"]].append(
                {"norm": normalize(body), "score": c.get("score") or 0}
            )
            n += 1
    logging.info(f"Loaded {n} scored comments across {len(by_post)} posts.")
    return by_post


def post_scores():
    scores = {}
    with open(REDDIT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                scores[row["id"].strip()] = int(row["score"])
            except (ValueError, KeyError):
                pass
    return scores


def attribute(title, comments):
    """Score stats for comments in a post that mention `title`."""
    nt = normalize(title)
    if len(nt) < MIN_TITLE_LEN:
        return {"n_mentions": 0, "score_sum": None, "score_max": None,
                "score_mean": None, "match_confidence": "ambiguous"}
    pat = title_pattern(nt)
    hits = [c["score"] for c in comments if pat.search(c["norm"])]
    if not hits:
        return {"n_mentions": 0, "score_sum": None, "score_max": None,
                "score_mean": None, "match_confidence": "no_match"}
    return {
        "n_mentions": len(hits),
        "score_sum": int(sum(hits)),
        "score_max": int(max(hits)),
        "score_mean": round(sum(hits) / len(hits), 2),
        "match_confidence": "ok",
    }


def main():
    comments_by_post = load_comments_by_post()
    pscore = post_scores()

    agg = defaultdict(lambda: {"title": None, "n_posts": 0, "score_sum": 0,
                               "agreement_sum": 0.0, "agreement_n": 0})
    n_edges = 0

    with open(DATASET_CSV, encoding="utf-8") as f, \
            open(OUT_DATASET, "w", encoding="utf-8") as fd, \
            open(OUT_EDGES, "w", encoding="utf-8") as fe:
        for row in csv.DictReader(f):
            pid = row["post_id"]
            titles = safe_list(row["verified_movies"])
            ids = safe_list(row["tmdb_ids"])
            comments = comments_by_post.get(pid, [])

            movie_rows = []
            for title, tid in zip(titles, ids):
                a = attribute(title, comments)
                movie_rows.append({"tmdb_id": tid, "title": title, **a})

            # Within-post agreement: score_sum normalized by the post's max.
            sums = [m["score_sum"] for m in movie_rows if m["score_sum"] is not None]
            top = max(sums) if sums else 0
            for m in movie_rows:
                m["agreement_norm"] = (
                    round(m["score_sum"] / top, 4)
                    if (m["score_sum"] is not None and top > 0) else None
                )
                fe.write(json.dumps({"post_id": pid, **m}, ensure_ascii=False) + "\n")
                n_edges += 1
                # movie-level aggregation
                g = agg[m["tmdb_id"]]
                g["title"] = m["title"]
                if m["score_sum"] is not None:
                    g["n_posts"] += 1
                    g["score_sum"] += m["score_sum"]
                    if m["agreement_norm"] is not None:
                        g["agreement_sum"] += m["agreement_norm"]
                        g["agreement_n"] += 1

            fd.write(json.dumps({
                "post_id": pid,
                "post_score": pscore.get(pid),
                "n_comments_fetched": len(comments),
                "total_comment_score": int(sum(c["score"] for c in comments)),
                "descriptors": safe_list(row["descriptors"]),
                "image_local_path": row["image_local_path"],
                "movies": movie_rows,
            }, ensure_ascii=False) + "\n")

    # movie-level consensus ranking
    agreement = []
    for tid, g in agg.items():
        agreement.append({
            "tmdb_id": tid,
            "title": g["title"],
            "n_posts_mentioned_with_score": g["n_posts"],
            "total_score": g["score_sum"],
            "mean_agreement_norm": round(g["agreement_sum"] / g["agreement_n"], 4)
            if g["agreement_n"] else None,
        })
    agreement.sort(key=lambda x: -(x["total_score"] or 0))
    with open(OUT_AGREEMENT, "w", encoding="utf-8") as f:
        json.dump(agreement, f, ensure_ascii=False, indent=2)

    ok = sum(1 for _ in open(OUT_EDGES, encoding="utf-8"))
    logging.info(f"Wrote {OUT_DATASET}, {OUT_EDGES} ({n_edges} edges), {OUT_AGREEMENT}.")

    # quick attribution coverage report
    conf = defaultdict(int)
    for line in open(OUT_EDGES, encoding="utf-8"):
        conf[json.loads(line)["match_confidence"]] += 1
    print("\n=== ATTRIBUTION COVERAGE (edges) ===")
    for k, v in sorted(conf.items(), key=lambda x: -x[1]):
        print(f"  {k:>12}: {v} ({100*v/n_edges:.1f}%)")
    print("\n=== TOP 15 MOST-AGREED MOVIES (by total upvote across posts) ===")
    for m in agreement[:15]:
        print(f"  {m['total_score']:>6}  {m['title']}  "
              f"({m['n_posts_mentioned_with_score']} posts, "
              f"mean-agreement {m['mean_agreement_norm']})")


if __name__ == "__main__":
    main()
