"""
Fetch TMDB backdrop paths for the browser probe pool.

The public probe shows pairs of movie BACKDROPS (scene stills from TMDB's
CDN, displayed with attribution) instead of Reddit-scraped images. This
script pulls backdrop_path for every well-supported movie in the web index
and writes frontend/public/engine/probe.json with rows aligned to
movies.json indices.

Usage:
    python pipeline/fetch_probe_backdrops.py
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOVIES_JSON = os.path.join(REPO_ROOT, "frontend", "public", "engine", "movies.json")
OUT_JSON = os.path.join(REPO_ROOT, "frontend", "public", "engine", "probe.json")

MIN_SUPPORT = 8   # probe images should come from confidently-placed movies
MIN_VOTES = 200   # and be real, recognizable productions
WORKERS = 20


def api_key():
    with open(os.path.join(REPO_ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("TMDB_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("TMDB_API_KEY not in .env")


def main():
    key = api_key()
    meta = json.load(open(MOVIES_JSON, encoding="utf-8"))
    candidates = [(i, m) for i, m in enumerate(meta["movies"])
                  if m["n"] >= MIN_SUPPORT and m["v"] >= MIN_VOTES]
    logging.info(f"Fetching backdrops for {len(candidates)} movies...")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {key}"
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=WORKERS))

    def fetch(item):
        i, m = item
        try:
            r = session.get(f"https://api.themoviedb.org/3/movie/{m['id']}", timeout=15)
            if r.status_code != 200:
                return None
            b = r.json().get("backdrop_path")
            return {"i": i, "b": b} if b else None
        except requests.RequestException:
            return None

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = [r for r in pool.map(fetch, candidates) if r]

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"pool": rows}, f)
    logging.info(f"Wrote {len(rows)} probe movies to {OUT_JSON} "
                 f"({os.path.getsize(OUT_JSON) / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
