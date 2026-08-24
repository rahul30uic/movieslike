"""
Merge TMDB details + OMDb ratings into the static file the detail modal loads.

  data/movie_details_tmdb.json  (overview, trailer, imdb_id, tmdb_rating)
  data/movie_omdb_ratings.json  (imdb_rating, rt_rating)   [optional, add later]
      -> frontend/public/engine/movie_details.json  { tmdb_id: {...} }

Null/empty fields are dropped to keep the payload small.

Usage:
    python pipeline/build_movie_details.py
"""

import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
TMDB = os.path.join(DATA_DIR, "movie_details_tmdb.json")
OMDB = os.path.join(DATA_DIR, "movie_omdb_ratings.json")
OUT = os.path.join(REPO, "frontend", "public", "engine", "movie_details.json")


def main():
    tmdb = json.load(open(TMDB))
    omdb = json.load(open(OMDB)) if os.path.exists(OMDB) else {}
    logging.info(f"{len(tmdb)} TMDB details, {len(omdb)} OMDb ratings.")

    out = {}
    for mid, d in tmdb.items():
        o = omdb.get(mid, {})
        rec = {}
        if d.get("overview"):
            rec["overview"] = d["overview"]
        if d.get("trailer"):
            rec["trailer"] = d["trailer"]
        if d.get("imdb_id"):
            rec["imdb_id"] = d["imdb_id"]
        if d.get("tmdb_rating"):
            rec["tmdb_rating"] = d["tmdb_rating"]
        if o.get("imdb_rating"):
            rec["imdb_rating"] = o["imdb_rating"]
        if o.get("rt_rating"):
            rec["rt_rating"] = o["rt_rating"]
        out[mid] = rec

    json.dump(out, open(OUT, "w"))
    nr = sum(1 for r in out.values() if r.get("imdb_rating"))
    logging.info(f"Wrote {len(out)} movies ({os.path.getsize(OUT)/1e6:.1f}MB); "
                 f"{nr} have IMDb/RT ratings -> {OUT}")


if __name__ == "__main__":
    main()
