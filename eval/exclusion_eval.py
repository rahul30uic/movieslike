"""
Exclusion-filter eval (#3): does the affect exclusion stop a query from
returning movies drenched in emotions it clearly didn't ask for?

Replicates the full browser serving path (query tower -> retrieve -> rerank)
with and without the exclusion penalty, on curated queries that carry a
human "must-not-contain" emotion. Reports the average forbidden-emotion score
in the top-N (lower is better) and the violation rate (top-N movies scoring
high on a forbidden emotion). A control horror query checks we don't nuke
wanted horror.

Usage:
    python eval/exclusion_eval.py
"""

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(REPO, "frontend", "public", "engine")

AVERSIVE = ["horror", "fear", "disgust", "anxiety", "empathic_pain", "awkwardness"]
EXCL_THRESH, EXCL_WEIGHT = 0.5, 1.3
N, K, MIN_VOTES = 12, 200, 500
PREFIX = "Represent this sentence for searching relevant passages: "

QUERIES = [
    ("cozy rainy night, gentle loneliness", ["horror", "fear"]),
    ("warm feel-good comedy for date night", ["horror", "fear", "disgust"]),
    ("peaceful calming nature, serene and slow", ["horror", "fear", "anxiety"]),
    ("nostalgic childhood wonder and awe", ["horror", "disgust"]),
    ("bittersweet melancholy, quietly sad", ["horror", "fear"]),
    ("terrifying edge-of-your-seat horror", []),  # control: keep horror
]


def load():
    movies = json.load(open(os.path.join(ENGINE, "tt_movies.json")))["movies"]
    dim = json.load(open(os.path.join(ENGINE, "tt_movies.json")))["dim"]
    items = np.fromfile(os.path.join(ENGINE, "tt_items.bin"), np.float16).astype(np.float32).reshape(len(movies), dim)
    wf = np.fromfile(os.path.join(ENGINE, "tt_query_tower.bin"), np.float16).astype(np.float32)
    sh = json.load(open(os.path.join(ENGINE, "tt_query_tower.json")))["shapes"]
    o = 0
    def take(n):
        nonlocal o; s = wf[o:o+n]; o += n; return s
    w1 = take(sh["w1"][0]*sh["w1"][1]).reshape(sh["w1"]); b1 = take(sh["b1"][0])
    w2 = take(sh["w2"][0]*sh["w2"][1]).reshape(sh["w2"]); b2 = take(sh["b2"][0])
    anc = json.load(open(os.path.join(ENGINE, "affect_anchors.json")))
    dims = anc["dims"]; anchors = np.array(anc["vecs"], np.float32)
    aff = np.fromfile(os.path.join(ENGINE, "affect.bin"), np.float16).astype(np.float32).reshape(len(movies), len(dims))
    return movies, items, (w1, b1, w2, b2), dims, anchors, aff


def main():
    movies, items, (w1, b1, w2, b2), dims, anchors, aff = load()
    aversive_idx = [dims.index(d) for d in AVERSIVE]
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="mps")
    gelu = lambda v: 0.5*v*(1+np.tanh(0.7978845608*(v+0.044715*v**3)))
    zc = lambda a: (a-a.mean())/(a.std()+1e-9)

    def run(query, exclude):
        qb = bge.encode(PREFIX+query, normalize_embeddings=True)
        tgt = np.concatenate([qb, np.zeros(768, np.float32)]); tgt /= np.linalg.norm(tgt)+1e-9
        z = gelu(w1@tgt+b1); z = w2@z+b2; z /= np.linalg.norm(z)+1e-9
        sims = items @ z
        cand = [i for i in np.argsort(-sims) if movies[i]["v"] >= MIN_VOTES][:K]
        base = zc(sims[cand]) - 0.3*zc(np.log1p([movies[i]["n"] for i in cand]))
        if exclude:
            s = anchors @ qb
            want = (s-s.min())/(np.ptp(s)+1e-9)
            for k, ci in enumerate(cand):
                clash = sum(aff[ci, d]*max(0, EXCL_THRESH-want[d]) for d in aversive_idx)
                base[k] -= EXCL_WEIGHT*clash
        base = (base-base.min())/(np.ptp(base)+1e-9)
        order = np.argsort(-base)[:N]
        return [cand[k] for k in order]

    print(f"\n=== EXCLUSION EVAL (top-{N}) — forbidden-emotion score & violations ===")
    print(f"{'query':<42}{'forbid':>16}{'off':>12}{'on':>12}")
    for query, forbid in QUERIES:
        fidx = [dims.index(f) for f in forbid]
        def score(res):
            if not fidx:  # control: report horror presence (want it kept)
                h = dims.index("horror")
                return np.mean([aff[i, h] for i in res])
            return np.mean([max(aff[i, f] for f in fidx) for i in res])
        off = score(run(query, False)); on = score(run(query, True))
        tag = "  (keep↑)" if not forbid else ""
        print(f"{query[:40]:<42}{','.join(forbid)[:15]:>16}{off:>12.3f}{on:>12.3f}{tag}")
    print("\nforbidden rows: lower 'on' = exclusion working. control row: 'on' should stay high.")


if __name__ == "__main__":
    main()
