"""
Build affect ANCHOR vectors for browser-side query parsing (#4).

Each kept emotion gets a short descriptive phrase, embedded with bge. In the
browser we project a text query onto these anchors (cosine) to read its affect
profile — "cozy rainy night" lights up calmness, "it's raining" aligns with
nothing and washes out. Pure vector math, no LLM in the query path.

Reads the kept dims from data/affect_meta.json.
Output: frontend/public/engine/affect_anchors.json  { dims, vecs (K x 768) }

Usage:
    python pipeline/build_affect_anchors.py
"""

import json
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(REPO, "data", "affect_meta.json")
OUT = os.path.join(REPO, "frontend", "public", "engine", "affect_anchors.json")

# short, mood-oriented phrases per emotion (the query is matched against these)
PHRASES = {
    "admiration": "admirable, inspiring, heroic, noble",
    "adoration": "tender adoration, cherished, devoted love",
    "aesthetic_appreciation": "beautiful, gorgeous, visually stunning, painterly",
    "amusement": "funny, playful, comedic, lighthearted",
    "anxiety": "anxious, tense, uneasy, nerve-wracking",
    "awe": "awe-inspiring, vast, sublime, breathtaking",
    "awkwardness": "awkward, cringe, uncomfortable, embarrassing",
    "boredom": "slow, dull, uneventful, boring",
    "calmness": "calm, cozy, peaceful, serene, gentle, soothing",
    "confusion": "confusing, disorienting, surreal, puzzling",
    "craving": "mouth-watering, tempting, indulgent craving",
    "disgust": "disgusting, gross, repulsive, revolting",
    "empathic_pain": "heartbreaking, painful, aching, sorrowful",
    "entrancement": "hypnotic, mesmerizing, dreamlike, entrancing",
    "excitement": "exciting, thrilling, high-energy, exhilarating",
    "fear": "scary, frightening, terror, menacing",
    "horror": "horrifying, nightmarish, dread, gruesome",
    "interest": "intriguing, absorbing, thought-provoking",
    "joy": "joyful, happy, uplifting, warm, feel-good",
    "nostalgia": "nostalgic, wistful, bittersweet memories, retro",
    "relief": "relieving, cathartic, reassuring",
    "romance": "romantic, tender love, intimate, yearning",
    "sadness": "sad, melancholy, mournful, lonely",
    "satisfaction": "satisfying, gratifying, resolved",
    "sexual_desire": "sensual, sultry, seductive, erotic",
    "sympathy": "sympathetic, compassionate, tender-hearted",
    "triumph": "triumphant, victorious, rousing, inspiring win",
}


def main():
    dims = json.load(open(META))["dims"]
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="mps")
    texts = [PHRASES[d] for d in dims]
    vecs = bge.encode(texts, normalize_embeddings=True).astype(np.float32)

    json.dump({"dims": dims,
               "vecs": [[round(float(x), 5) for x in row] for row in vecs]},
              open(OUT, "w"))
    logging.info(f"Wrote {len(dims)} affect anchors ({vecs.shape[1]}-d) -> {OUT}")


if __name__ == "__main__":
    main()
