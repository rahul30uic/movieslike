"""
Turn the LLM-labeled core affect scores into an affect vector for EVERY movie.

1. Prune the 27 emotions down to the ones that actually vary across movies
   (a dimension every film scores the same on carries no signal).
2. Train a content model (ridge regression) that predicts the pruned affect
   vector from a movie's facts embedding (bge of title/genres/overview), so
   the ~12k movies outside the labeled core also get an affect profile.
3. Report how well each dimension is predictable (so we know how much to trust
   the affect prior on the tail — used by the shrinkage step).

Inputs : data/movie_affect_core.json, data/movie_facts_vectors.npz
Outputs: data/movie_affect.json        { tmdb_id: [scores...] }  (all movies)
         data/affect_meta.json         { dims: [...], predictable: {dim: r2} }

Usage:
    python pipeline/build_affect_vectors.py
"""

import json
import logging
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
LABELS = os.path.join(DATA_DIR, "movie_affect_core.json")
FACTS = os.path.join(DATA_DIR, "movie_facts_vectors.npz")
OUT = os.path.join(DATA_DIR, "movie_affect.json")
META = os.path.join(DATA_DIR, "affect_meta.json")

EMOTIONS = [
    "admiration", "adoration", "aesthetic_appreciation", "amusement", "anxiety",
    "awe", "awkwardness", "boredom", "calmness", "confusion", "craving", "disgust",
    "empathic_pain", "entrancement", "excitement", "fear", "horror", "interest",
    "joy", "nostalgia", "relief", "romance", "sadness", "satisfaction",
    "sexual_desire", "sympathy", "triumph",
]
MIN_STD = 0.12   # prune emotions that barely vary across the core


def main():
    labels = {int(k): v for k, v in json.load(open(LABELS)).items()}
    fz = np.load(FACTS)
    facts = {int(t): v for t, v in zip(fz["tmdb_ids"], fz["vectors"].astype(np.float32))}
    logging.info(f"{len(labels)} labeled core movies, {len(facts)} with facts vectors.")

    core_ids = [t for t in labels if t in facts]
    Y = np.array([[labels[t][e] for e in EMOTIONS] for t in core_ids], dtype=np.float32)

    # 1. prune low-variance dims
    stds = Y.std(0)
    keep = [i for i, s in enumerate(stds) if s >= MIN_STD]
    dims = [EMOTIONS[i] for i in keep]
    logging.info(f"kept {len(dims)}/{len(EMOTIONS)} dims: {dims}")
    logging.info("pruned (low variance): "
                 + ", ".join(EMOTIONS[i] for i in range(len(EMOTIONS)) if i not in keep))
    Yk = Y[:, keep]

    # 2. content model: facts embedding -> affect (with CV R^2 per dim)
    X = np.array([facts[t] for t in core_ids], dtype=np.float32)
    r2 = {}
    kf = KFold(5, shuffle=True, random_state=42)
    preds = np.zeros_like(Yk)
    for tr, te in kf.split(X):
        m = Ridge(alpha=10.0).fit(X[tr], Yk[tr])
        preds[te] = m.predict(X[te])
    for j, d in enumerate(dims):
        ss_res = ((Yk[:, j] - preds[:, j]) ** 2).sum()
        ss_tot = ((Yk[:, j] - Yk[:, j].mean()) ** 2).sum()
        r2[d] = round(float(1 - ss_res / (ss_tot + 1e-9)), 3)
    logging.info("content-model R^2 per dim (how well facts predict affect):")
    for d, v in sorted(r2.items(), key=lambda x: -x[1]):
        logging.info(f"    {d:24} {v}")

    # fit on all core, predict everyone
    model = Ridge(alpha=10.0).fit(X, Yk)
    out = {}
    for tid, fv in facts.items():
        if tid in labels:  # keep the real label for the core
            vec = [labels[tid][EMOTIONS[i]] for i in keep]
        else:
            vec = np.clip(model.predict(fv[None])[0], 0, 1).tolist()
        out[tid] = [round(float(x), 4) for x in vec]

    json.dump(out, open(OUT, "w"))
    json.dump({"dims": dims, "predictable_r2": r2, "n_core": len(core_ids)},
              open(META, "w"), indent=2)
    logging.info(f"Wrote {len(out)} affect vectors ({len(dims)}-d) -> {OUT}")


if __name__ == "__main__":
    main()
