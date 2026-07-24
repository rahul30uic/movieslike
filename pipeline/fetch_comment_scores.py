"""
Fetch per-comment upvote scores from the Arctic Shift Reddit archive and join
them to our posts. Our original scrape kept comment author+body but dropped
scores; this backfills them.

For each post id in Reddit_data.csv, query Arctic Shift's comments API by
link_id (t3_<id>), paginate over created_utc, and write one row per comment
with its score. Resumable: posts already in the output are skipped.

Output: data/comments_with_scores.jsonl
    {post_id, id, parent_id, author, body, score, ups, downs,
     controversiality, created_utc}

Usage:
    python pipeline/fetch_comment_scores.py            # all posts
    python pipeline/fetch_comment_scores.py --limit 20 # smoke test
"""

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
INPUT_CSV = os.path.join(DATA_DIR, "Reddit_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "comments_with_scores.jsonl")

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
HEADERS = {"User-Agent": "movieslike-research/1.0 (personal ML project)"}
PAGE = 100
MAX_PAGES = 8          # safety cap; deepest posts have a few hundred comments
WORKERS = 4
RPS = 8.0             # global request budget — be polite to a free service
KEEP = ("id", "parent_id", "author", "body", "score", "ups", "downs",
        "controversiality", "created_utc")


class RateLimiter:
    def __init__(self, rps):
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.next_slot = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            slot = max(self.next_slot, now)
            self.next_slot = slot + self.interval
        time.sleep(max(0.0, slot - time.monotonic()))


def post_ids():
    seen, ids = set(), []
    with open(INPUT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["id"].strip()
            if pid and pid not in seen:
                seen.add(pid)
                ids.append(pid)
    return ids


def already_done():
    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["post_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def fetch_post(pid, limiter):
    """All comments for one post, paginated over created_utc with dedup."""
    out, seen = [], set()
    after = None
    for _ in range(MAX_PAGES):
        params = {"link_id": f"t3_{pid}", "limit": PAGE, "sort": "asc"}
        if after is not None:
            params["after"] = after
        for attempt in range(4):
            try:
                limiter.wait()
                r = requests.get(API, params=params, headers=HEADERS, timeout=30)
                if r.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json().get("data", [])
                break
            except requests.RequestException:
                if attempt == 3:
                    return out  # give up on this post, keep what we have
                time.sleep(1.5 * (attempt + 1))
        else:
            break

        new = [c for c in data if c.get("id") not in seen]
        for c in new:
            seen.add(c["id"])
            row = {"post_id": pid, **{k: c.get(k) for k in KEEP}}
            out.append(row)
        if len(data) < PAGE or not new:
            break
        after = max(c.get("created_utc", 0) for c in data)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ids = post_ids()
    done = already_done()
    todo = [p for p in ids if p not in done]
    if args.limit:
        todo = todo[:args.limit]
    logging.info(f"{len(ids)} posts total; {len(done)} done; fetching {len(todo)}.")
    if not todo:
        logging.info("Nothing to fetch.")
        return

    limiter = RateLimiter(RPS)
    lock = threading.Lock()
    n_posts = n_comments = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {pool.submit(fetch_post, p, limiter): p for p in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                rows = fut.result()
                with lock:
                    for row in rows:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    n_posts += 1
                    n_comments += len(rows)
                if i % 100 == 0:
                    logging.info(f"{i}/{len(todo)} posts | {n_comments} comments so far")

    logging.info(f"Done. {n_posts} posts, {n_comments} comments -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
