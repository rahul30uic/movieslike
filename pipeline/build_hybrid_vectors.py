"""
Build the production post vectors in the winning hybrid architecture:

    post_vector = normalize([ w_c * bge(caption) ; (1 - w_c) * siglip(images) ])

(1536-dim; see eval_results/hybrid_comparison.json — rare lift 6.77 vs 5.91
for the best single-space alternative.) Posts without an image get a zero
image block; posts without a caption are dropped (they had no content at all).

Output schema matches posts_with_vectors.json so the eval harness and
build_movie_vectors.py consume it unchanged.

Usage:
    python build_hybrid_vectors.py            # writes posts_with_hybrid_vectors.json
    python build_hybrid_vectors.py --eval     # also score it
"""

import argparse
import json
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
CAPTION_VECS = os.path.join(DATA_DIR, "posts_with_caption_vectors.json")
NPZ_FILE = os.path.join(DATA_DIR, "post_modality_vectors.npz")
META_FILE = os.path.join(DATA_DIR, "post_modality_meta.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "posts_with_hybrid_vectors.json")

# Recalibrated for the LoRA-tuned image tower: contrastive fine-tuning
# sharpened image-block similarities, shifting the optimal fusion from
# 0.5 (frozen) to 0.3 (swept on full-corpus eval; 8.10 vs 7.51 lift).
CAPTION_WEIGHT = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--image-npz", default=NPZ_FILE,
                    help="Image-vector source (e.g. post_modality_vectors_lora.npz)")
    args = ap.parse_args()

    data = np.load(args.image_npz)
    iv, hi = data["image_vecs"], data["has_image"]
    meta = [json.loads(l) for l in open(META_FILE, encoding="utf-8")]
    row_of = {m["post_id"]: i for i, m in enumerate(meta)}

    n_out = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for line in open(CAPTION_VECS, encoding="utf-8"):
            p = json.loads(line)
            i = row_of.get(p["post_id"])
            if i is None:
                continue
            c = np.asarray(p["combined_vector"], dtype=np.float32)
            c = c / np.linalg.norm(c)
            img = iv[i] if hi[i] else np.zeros_like(c)
            v = np.concatenate([CAPTION_WEIGHT * c, (1 - CAPTION_WEIGHT) * img])
            v = v / np.linalg.norm(v)
            out.write(json.dumps({
                "post_id": p["post_id"],
                "descriptors": p["descriptors"],
                "tmdb_ids": p["tmdb_ids"],
                "image_exists": bool(hi[i]),
                "caption": p["caption"],
                "combined_vector": [round(float(x), 6) for x in v],
            }) + "\n")
            n_out += 1
    logging.info(f"Wrote {n_out} hybrid post vectors (1536-dim) to {OUTPUT_FILE}.")

    if args.eval:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), "eval"))
        from eval_embeddings import evaluate
        posts = [json.loads(l) for l in open(OUTPUT_FILE, encoding="utf-8")]
        posts = [p for p in posts if p["tmdb_ids"]]
        r = evaluate(posts)
        print(json.dumps({k: r[k] for k in ("n_posts", "knn", "by_modality")}, indent=2))
        with open(os.path.join(os.path.dirname(SCRIPT_DIR), "eval", "eval_results", "hybrid_full_corpus.json"), "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
