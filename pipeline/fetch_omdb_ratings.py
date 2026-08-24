"""
Fetch IMDb + Rotten Tomatoes ratings from OMDb, by imdb_id.

Needs OMDB_API_KEY in .env. Free tier = 1,000/day (this set is ~5k, so use the
$1/mo Patreon key = 100k/day, or run over several days — it's resumable).

Reads the imdb_ids from data/movie_details_tmdb.json.
Output: data/movie_omdb_ratings.json  { tmdb_id: {imdb_rating, rt_rating} }

Then re-run pipeline/build_movie_details.py to merge into the served file.

Usage:
    python pipeline/fetch_omdb_ratings.py
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
TMDB = os.path.join(DATA_DIR, "movie_details_tmdb.json")
OUT = os.path.join(DATA_DIR, "movie_omdb_ratings.json")
WORKERS = 10


def api_key():
    for path in (os.path.join(REPO, ".env"), os.path.join(REPO, "movie extraction", ".env")):
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                if line.startswith("OMDB_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError("OMDB_API_KEY not in .env — get a free key at omdbapi.com")


def rt_from(ratings):
    for r in ratings or []:
        if r.get("Source") == "Rotten Tomatoes":
            return r.get("Value")  # e.g. "94%"
    return None


def main():
    key = api_key()
    tmdb = json.load(open(TMDB))
    done = {}
    if os.path.exists(OUT):
        done = json.load(open(OUT))
    todo = [(mid, d["imdb_id"]) for mid, d in tmdb.items()
            if d.get("imdb_id") and mid not in done]
    logging.info(f"{len(done)} already fetched; {len(todo)} to go.")

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=WORKERS))
    lock = threading.Lock()

    def fetch(item):
        mid, imdb = item
        try:
            r = session.get("https://www.omdbapi.com/",
                            params={"apikey": key, "i": imdb, "tomatoes": "true"}, timeout=15)
            if r.status_code != 200:
                return mid, None
            d = r.json()
            if d.get("Response") != "True":
                return mid, {"_err": d.get("Error", "")}
            imdb_rating = d.get("imdbRating")
            return mid, {
                "imdb_rating": imdb_rating if imdb_rating and imdb_rating != "N/A" else None,
                "rt_rating": rt_from(d.get("Ratings")),
            }
        except requests.RequestException:
            return mid, None

    n_ok, quota_hit = 0, False
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, it) for it in todo]
        for j, fut in enumerate(as_completed(futures), 1):
            mid, res = fut.result()
            if res and "_err" in res:
                if "limit" in res["_err"].lower():
                    quota_hit = True
                continue
            if res:
                with lock:
                    done[mid] = res
                    n_ok += 1
            if j % 500 == 0:
                logging.info(f"{j}/{len(todo)} ({n_ok} ok)")
                with lock:
                    json.dump(done, open(OUT, "w"))

    json.dump(done, open(OUT, "w"))
    rt = sum(1 for v in done.values() if v.get("rt_rating"))
    logging.info(f"Done. {len(done)} movies — {rt} with RT ratings -> {OUT}")
    if quota_hit:
        logging.warning("Hit the OMDb daily quota — re-run tomorrow (resumable), "
                        "or use the $1/mo Patreon key (100k/day).")


if __name__ == "__main__":
    main()
