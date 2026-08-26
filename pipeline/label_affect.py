"""
Label each core movie with Cowen & Keltner's 27 affect dimensions (0-1),
using an LLM over the movie's plot + genres + keywords + the mood-captions of
the posts that recommended it (the crowd's own description of how it feels).

This gives every well-supported movie a NAMED emotional profile — a signed,
interpretable representation the embedding lacks. It's the keystone for the
exclusion filter, query parsing, and tail shrinkage.

Core = support >= 5 posts. Resumable.
Output: data/movie_affect_core.json  { tmdb_id: {emotion: score, ...} }

Usage:
    python pipeline/label_affect.py --limit 15    # sample
    python pipeline/label_affect.py               # full core
"""

import argparse
import json
import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
MOVIES = os.path.join(DATA_DIR, "movie_vectors_hybrid.json")
POSTS = os.path.join(DATA_DIR, "posts_with_hybrid_vectors.json")
KEYWORDS = os.path.join(DATA_DIR, "movie_keywords.json")
CACHE_DB = os.path.join(DATA_DIR, "tmdb_cache.sqlite")
OUT = os.path.join(DATA_DIR, "movie_affect_core.json")

MODEL = "gemini-2.5-flash"
MIN_SUPPORT = 5
MAX_CAPTIONS = 8
RPM = 150
WORKERS = 8

EMOTIONS = [
    "admiration", "adoration", "aesthetic_appreciation", "amusement", "anxiety",
    "awe", "awkwardness", "boredom", "calmness", "confusion", "craving", "disgust",
    "empathic_pain", "entrancement", "excitement", "fear", "horror", "interest",
    "joy", "nostalgia", "relief", "romance", "sadness", "satisfaction",
    "sexual_desire", "sympathy", "triumph",
]

PROMPT = (
    "You rate the FELT EXPERIENCE of watching a film across 27 emotions. For each, "
    "give 0.0-1.0 for how strongly it characterizes the movie's mood/atmosphere "
    "(not merely whether it appears in the plot). Most films are strong on only a "
    "few; use low scores freely. Base it on the vibe, using the audience mood notes "
    "when given. Output ONLY a JSON object with exactly these keys:\n"
    + ", ".join(EMOTIONS)
)


class RateLimiter:
    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.next = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            slot = max(self.next, now)
            self.next = slot + self.interval
        time.sleep(max(0.0, slot - time.monotonic()))


def load_key():
    for p in (os.path.join(REPO, ".env"), os.path.join(REPO, "movie extraction", ".env")):
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("GEMINI_API_KEY not found")


def load_context():
    meta = {}
    conn = sqlite3.connect(CACHE_DB)
    for (resp,) in conn.execute("SELECT response FROM tmdb_cache"):
        try:
            for r in (json.loads(resp).get("results") or []):
                if r.get("id") and r["id"] not in meta:
                    meta[r["id"]] = {"title": r.get("title") or r.get("name") or "",
                                     "year": (r.get("release_date") or "")[:4],
                                     "overview": (r.get("overview") or "")[:400]}
        except json.JSONDecodeError:
            continue
    conn.close()
    kw = {int(k): v for k, v in json.load(open(KEYWORDS)).items()}
    caps = defaultdict(list)
    for line in open(POSTS, encoding="utf-8"):
        p = json.loads(line)
        if p.get("caption"):
            for t in set(p.get("tmdb_ids", [])):
                caps[t].append(p["caption"])
    return meta, kw, caps


def context_text(tid, m, kw, caps):
    parts = [f"{m.get('title','')} ({m.get('year','')})".strip()]
    ks = kw.get(tid, [])
    if ks:
        parts.append("Tags: " + ", ".join(ks[:15]))
    if m.get("overview"):
        parts.append("Plot: " + m["overview"])
    cs = caps.get(tid, [])[:MAX_CAPTIONS]
    if cs:
        parts.append("Audience mood notes:\n- " + "\n- ".join(c[:200] for c in cs))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    core = [json.loads(l) for l in open(MOVIES, encoding="utf-8")]
    core = [m for m in core if m["n_posts"] >= MIN_SUPPORT]
    meta, kw, caps = load_context()

    done = {}
    if os.path.exists(OUT):
        done = {int(k): v for k, v in json.load(open(OUT)).items()}
    todo = [m for m in core if m["tmdb_id"] not in done and m["tmdb_id"] in meta]
    if args.limit:
        todo = todo[:args.limit]
    logging.info(f"core={len(core)}, {len(done)} labeled, {len(todo)} to go.")

    client = genai.Client(api_key=load_key(), http_options={"timeout": 90_000})
    cfg = types.GenerateContentConfig(response_mime_type="application/json",
                                      thinking_config=types.ThinkingConfig(thinking_budget=0))
    limiter = RateLimiter(RPM)
    lock = threading.Lock()

    def label(m):
        tid = m["tmdb_id"]
        prompt = PROMPT + "\n\n" + context_text(tid, meta[tid], kw, caps)
        for attempt in range(5):
            try:
                limiter.wait()
                r = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
                d = json.loads(r.text)
                scores = {e: float(max(0.0, min(1.0, d.get(e, 0.0)))) for e in EMOTIONS}
                return tid, scores
            except Exception as e:
                if attempt == 4:
                    logging.error(f"{tid}: {str(e)[:80]}")
                    return tid, None
                time.sleep(2 * (attempt + 1))
        return tid, None

    n_ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(label, m): m for m in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            tid, scores = fut.result()
            if scores:
                with lock:
                    done[tid] = scores
                    n_ok += 1
            if i % 200 == 0:
                logging.info(f"{i}/{len(todo)} ({n_ok} ok)")
                with lock:
                    json.dump(done, open(OUT, "w"))

    json.dump(done, open(OUT, "w"))
    logging.info(f"Done. {len(done)} movies labeled -> {OUT}")


if __name__ == "__main__":
    main()
