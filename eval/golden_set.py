"""
Human golden-set evaluation: 25 head-space scenarios -> top-5 movies each,
written to a rating sheet a human scores in ~20 minutes.

Metric: precision@5 = fraction of recommendations rated "yes, I'd genuinely
consider watching that tonight" for the scenario. This is the human-judgment
complement to the movie-overlap proxy metric.

Usage:
    python eval/golden_set.py          # writes eval/golden_set_ratings.md
    python eval/golden_set.py --score  # parses the filled-in sheet
"""

import argparse
import json
import logging
import os
import re

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MOVIES_FILE = os.path.join(DATA_DIR, "movie_vectors_hybrid.json")
SHEET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set_ratings.md")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results", "golden_set.json")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ALPHA, MIN_VOTES, MIN_SUPPORT, TOP_N = 0.3, 500, 3, 5

SCENARIOS = [
    "cozy rainy night, gentle loneliness, warm lamplight",
    "neon-lit city at 3am, wide awake and a little lost",
    "slow-burn dread in a small town where everyone smiles too much",
    "sun-bleached desert highway, dust and freedom",
    "characters who are completely unhinged, chaotic energy",
    "quiet grief, long silences, healing by the sea",
    "snowed-in cabin, candlelight, something creaking upstairs",
    "youthful summer nostalgia, bikes at dusk, nothing lasts",
    "opulent ballrooms, whispered betrayals, velvet and candlewax",
    "gritty rain-soaked crime, tired detectives, moral fog",
    "cosmic wonder, feeling tiny under an enormous sky",
    "deadpan absurd humor, everyone acts normal but nothing is",
    "foggy coastal village, salt air, an old secret",
    "adrenaline and engines, everything chrome and fast",
    "melancholy autumn campus, sweaters, unspoken feelings",
    "surreal dream logic, doors that lead somewhere impossible",
    "warm found-family dinner scenes, bittersweet goodbyes",
    "paranoid 70s conspiracy, wiretaps and beige hallways",
    "lush jungle expedition, awe and menace in equal parts",
    "lonely astronaut energy, hum of machines, distant Earth",
    "small kindnesses between strangers on a night train",
    "glitter and ruin, fame curdling into isolation",
    "medieval mud and iron, torchlight, grim resolve",
    "sticky southern summer, secrets on the porch",
    "rebellious teens against a dying industrial town",
]


def load_movies():
    metas, vecs = [], []
    for line in open(MOVIES_FILE, encoding="utf-8"):
        m = json.loads(line)
        vecs.append(m.pop("vector"))
        metas.append(m)
    V = np.asarray(vecs, dtype=np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return metas, V


def rank(metas, V, target):
    sims = V @ (target / (np.linalg.norm(target) + 1e-9))
    z = (sims - sims.mean()) / (sims.std() + 1e-9)
    lp = np.log2(1 + np.array([m["n_posts"] for m in metas]))
    pen = (lp - lp.mean()) / (lp.std() + 1e-9)
    score = z - ALPHA * pen
    for i, m in enumerate(metas):
        if m["n_posts"] < MIN_SUPPORT or (m.get("vote_count") or 0) < MIN_VOTES:
            score[i] = -np.inf
    top = np.argsort(-score)[:TOP_N]
    return [metas[i]["title"] for i in top]


def generate():
    from sentence_transformers import SentenceTransformer
    metas, V = load_movies()
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    dim = V.shape[1]

    lines = [
        "# Golden-set rating sheet",
        "",
        "For each movie: does it genuinely fit the mood — would you consider",
        "watching it on that night? Mark `[y]` for yes, `[n]` for no.",
        "Rate the FIT, not whether you personally love the film.",
        "",
    ]
    for k, scenario in enumerate(SCENARIOS, 1):
        q = bge.encode(QUERY_PREFIX + scenario, normalize_embeddings=True)
        target = np.concatenate([q, np.zeros(dim - len(q), dtype=np.float32)])
        titles = rank(metas, V, target)
        lines.append(f"## {k}. \"{scenario}\"")
        for t in titles:
            lines.append(f"- [ ] {t}")
        lines.append("")
    with open(SHEET, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"Wrote rating sheet: {SHEET}")


def score():
    text = open(SHEET, encoding="utf-8").read()
    marks = re.findall(r"^- \[([ynYN])\]", text, flags=re.M)
    if not marks:
        logging.error("No [y]/[n] marks found — fill in the sheet first.")
        return
    y = sum(1 for m in marks if m.lower() == "y")
    result = {"rated": len(marks), "yes": y, "precision_at_5": round(y / len(marks), 3)}
    with open(RESULTS, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    if ap.parse_args().score:
        score()
    else:
        generate()
