"""
Fetch per-movie detail (overview, YouTube trailer, imdb_id) from TMDB for the
recommendable set (votes >= MIN_VOTES), for the movie detail modal.

OMDb IMDb/RT ratings are merged in later by fetch_omdb_ratings.py (needs a key).

Output: data/movie_details_tmdb.json
    { tmdb_id: {overview, trailer (yt key), imdb_id, tmdb_rating} }

Usage:
    python pipeline/fetch_movie_details.py
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
MOVIES_FILE = os.path.join(REPO, "frontend", "public", "engine", "tt_movies.json")
OUT = os.path.join(DATA_DIR, "movie_details_tmdb.json")
MIN_VOTES = 500
WORKERS = 24


def api_key():
    with open(os.path.join(REPO, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("TMDB_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("TMDB_API_KEY not in .env")


def pick_trailer(videos):
    vids = (videos or {}).get("results", [])
    yt = [v for v in vids if v.get("site") == "YouTube"]
    for want in ("Trailer", "Teaser"):
        for v in yt:
            if v.get("type") == want and v.get("official"):
                return v["key"]
        for v in yt:
            if v.get("type") == want:
                return v["key"]
    return yt[0]["key"] if yt else None


def main():
    movies = json.load(open(MOVIES_FILE))["movies"]
    ids = [m["id"] for m in movies if m["v"] >= MIN_VOTES]
    done = {}
    if os.path.exists(OUT):
        done = {int(k): v for k, v in json.load(open(OUT)).items()}
    todo = [i for i in ids if i not in done]
    logging.info(f"{len(done)} already fetched; {len(todo)} recommendable movies to go.")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {api_key()}"
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=WORKERS))
    lock = threading.Lock()

    def fetch(mid):
        for kind in ("movie", "tv"):
            try:
                r = session.get(
                    f"https://api.themoviedb.org/3/{kind}/{mid}",
                    params={"append_to_response": "videos,external_ids"}, timeout=15)
                if r.status_code != 200:
                    continue
                d = r.json()
                return mid, {
                    "overview": (d.get("overview") or "").strip(),
                    "trailer": pick_trailer(d.get("videos")),
                    "imdb_id": d.get("imdb_id") or (d.get("external_ids") or {}).get("imdb_id"),
                    "tmdb_rating": round(d.get("vote_average") or 0, 1),
                }
            except requests.RequestException:
                continue
        return mid, None

    n_ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, i) for i in todo]
        for j, fut in enumerate(as_completed(futures), 1):
            mid, det = fut.result()
            if det:
                with lock:
                    done[mid] = det
                    n_ok += 1
            if j % 1000 == 0:
                logging.info(f"{j}/{len(todo)} ({n_ok} ok)")
                with lock:
                    json.dump(done, open(OUT, "w"))

    json.dump(done, open(OUT, "w"))
    tr = sum(1 for v in done.values() if v.get("trailer"))
    im = sum(1 for v in done.values() if v.get("imdb_id"))
    logging.info(f"Done. {len(done)} movies — {tr} with trailers, {im} with imdb_id -> {OUT}")


if __name__ == "__main__":
    main()
