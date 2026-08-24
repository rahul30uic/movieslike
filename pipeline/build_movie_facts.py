"""
Build movie "facts" vectors from TMDB metadata (the cold-start half of the
hybrid item tower).

Every movie — even one Reddit barely mentioned — has TMDB metadata, so a
facts vector gives the long tail a real representation instead of a vector
built from one noisy post. We compose a short factual blurb per movie
(title, year, genres, overview) and embed it with bge, the same encoder and
space as the vibe captions.

Source: data/tmdb_cache.sqlite (94k cached TMDB responses from extraction).
Output: data/movie_facts_vectors.npz  (tmdb_ids, vectors [N,768])

Usage:
    python pipeline/build_movie_facts.py
"""

import json
import logging
import os
import sqlite3

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
CACHE_DB = os.path.join(DATA_DIR, "tmdb_cache.sqlite")
MOVIES_FILE = os.path.join(DATA_DIR, "movie_vectors_hybrid.json")
KEYWORDS_FILE = os.path.join(DATA_DIR, "movie_keywords.json")
OUT = os.path.join(DATA_DIR, "movie_facts_vectors.npz")
OUT_VIBE = os.path.join(DATA_DIR, "movie_vibefacts_vectors.npz")

GENRES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance",
    878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War",
    37: "Western", 10759: "Action & Adventure", 10762: "Kids", 10763: "News",
    10764: "Reality", 10765: "Sci-Fi & Fantasy", 10766: "Soap", 10767: "Talk",
    10768: "War & Politics",
}


def collect_metadata():
    """tmdb_id -> {title, year, genres, overview} from every cached result."""
    meta = {}
    conn = sqlite3.connect(CACHE_DB)
    for (response,) in conn.execute("SELECT response FROM tmdb_cache"):
        try:
            results = json.loads(response).get("results") or []
        except json.JSONDecodeError:
            continue
        for r in results:
            tid = r.get("id")
            if tid is None or tid in meta:
                continue
            date = r.get("release_date") or r.get("first_air_date") or ""
            meta[tid] = {
                "title": r.get("title") or r.get("name") or "",
                "year": date[:4],
                "genres": [GENRES.get(g) for g in (r.get("genre_ids") or []) if g in GENRES],
                "overview": (r.get("overview") or "").strip(),
            }
    conn.close()
    logging.info(f"Collected metadata for {len(meta)} tmdb ids from cache.")
    return meta


def facts_text(m):
    head = m["title"]
    if m["year"]:
        head += f" ({m['year']})"
    if m["genres"]:
        head += ". " + ", ".join(m["genres"])
    body = m["overview"]
    return f"{head}. {body}".strip()


def vibefacts_text(m, keywords):
    """Vibe register: genres + TMDB keyword tags, NO plot summary. Meant to
    match how queries are written ('foggy small-town dread') rather than plot."""
    parts = []
    if m["genres"]:
        parts.append(", ".join(m["genres"]))
    if keywords:
        parts.append(", ".join(keywords[:20]))
    return ". ".join(parts).strip()


def main():
    want = [json.loads(l)["tmdb_id"] for l in open(MOVIES_FILE, encoding="utf-8")]
    meta = collect_metadata()

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vibe", action="store_true",
                    help="Build genres+keywords 'vibe facts' (no plot) -> movie_vibefacts_vectors.npz")
    args, _ = ap.parse_known_args()

    keywords = {}
    if args.vibe:
        keywords = {int(k): v for k, v in json.load(open(KEYWORDS_FILE)).items()}

    ids, texts = [], []
    for tid in want:
        m = meta.get(tid)
        if not m:
            continue
        if args.vibe:
            txt = vibefacts_text(m, keywords.get(tid, []))
            if not txt:
                continue
        else:
            if not (m["overview"] or m["genres"]):
                continue
            txt = facts_text(m)
        ids.append(tid)
        texts.append(txt)
    logging.info(f"{len(ids)}/{len(want)} movies have usable "
                 f"{'vibe-facts' if args.vibe else 'facts'} "
                 f"({100*len(ids)/len(want):.1f}% coverage).")

    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="mps")
    vecs = bge.encode(texts, batch_size=128, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=True)

    out = OUT_VIBE if args.vibe else OUT
    np.savez_compressed(out, tmdb_ids=np.asarray(ids, dtype=np.int64),
                        vectors=vecs.astype(np.float16))
    logging.info(f"Wrote {len(ids)} vectors ({vecs.shape[1]}-dim) to {out}.")
    print("\nsample blurbs:")
    for t in texts[:4]:
        print("  -", t[:130])


if __name__ == "__main__":
    main()
