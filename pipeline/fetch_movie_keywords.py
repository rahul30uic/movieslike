"""
Fetch TMDB keywords for every movie in the catalog.

Keywords ("dystopia", "loneliness", "neon", "small town") are far closer to
the *mood* register our queries are written in than plot summaries are — so
they're the candidate cold-start feature for the two-tower item side.

Tries the /movie/{id}/keywords endpoint, falls back to /tv/{id}/keywords.
Resumable: ids already in the output file are skipped.

Output: data/movie_keywords.json  {tmdb_id: [keyword, ...]}

Usage:
    python pipeline/fetch_movie_keywords.py
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
FACTS_FILE = os.path.join(DATA_DIR, "movie_facts_vectors.npz")
OUT = os.path.join(DATA_DIR, "movie_keywords.json")
WORKERS = 24


def api_key():
    with open(os.path.join(REPO, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("TMDB_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("TMDB_API_KEY not in .env")


def main():
    ids = [int(t) for t in np.load(FACTS_FILE)["tmdb_ids"]]
    done = {}
    if os.path.exists(OUT):
        done = {int(k): v for k, v in json.load(open(OUT)).items()}
    todo = [i for i in ids if i not in done]
    logging.info(f"{len(done)} already fetched; {len(todo)} to go.")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {api_key()}"
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=WORKERS))
    lock = threading.Lock()

    def fetch(mid):
        for kind, key in (("movie", "keywords"), ("tv", "results")):
            try:
                r = session.get(f"https://api.themoviedb.org/3/{kind}/{mid}/keywords", timeout=15)
                if r.status_code == 200:
                    kws = [k["name"] for k in (r.json().get(key) or [])]
                    return mid, kws
            except requests.RequestException:
                continue
        return mid, None

    n_ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, i) for i in todo]
        for j, fut in enumerate(as_completed(futures), 1):
            mid, kws = fut.result()
            if kws is not None:
                with lock:
                    done[mid] = kws
                    n_ok += 1
            if j % 1000 == 0:
                logging.info(f"{j}/{len(todo)} fetched ({n_ok} with keywords)")
                with lock:
                    json.dump(done, open(OUT, "w"))

    json.dump(done, open(OUT, "w"))
    with_kw = sum(1 for v in done.values() if v)
    logging.info(f"Done. {len(done)} movies, {with_kw} have keywords -> {OUT}")


if __name__ == "__main__":
    main()
