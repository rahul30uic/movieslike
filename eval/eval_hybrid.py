"""
Three-way embedding comparison on the SAME post population:

  A. SigLIP fusion at the sweep-winning weight (w_text=0.1)
  B. bge-embedded VLM captions alone
  C. Hybrid: [w_c * caption_vec ; (1-w_c) * image_vec] concatenation, swept.
     With both blocks unit-norm, cosine on the concat equals the weighted sum
     of per-block cosines — image similarity from SigLIP, semantic similarity
     from captions, no cross-space alignment needed. Posts without an image
     get a zero image block.

Population: posts that have a caption AND >= 1 tmdb_id (the caption run's
coverage), so all rows are directly comparable.

Usage:
    python eval_hybrid.py
"""

import json
import logging
import os

import numpy as np

from eval_embeddings import evaluate, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
NPZ_FILE = os.path.join(DATA_DIR, "post_modality_vectors.npz")
META_FILE = os.path.join(DATA_DIR, "post_modality_meta.json")
CAPTION_VECS = os.path.join(DATA_DIR, "posts_with_caption_vectors.json")

SIGLIP_TEXT_W = 0.1
CAPTION_WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]


def main():
    data = np.load(NPZ_FILE)
    tv, iv = data["text_vecs"], data["image_vecs"]
    ht, hi = data["has_text"], data["has_image"]
    meta = [json.loads(l) for l in open(META_FILE, encoding="utf-8")]
    row_of = {m["post_id"]: i for i, m in enumerate(meta)}

    cap_posts = [json.loads(l) for l in open(CAPTION_VECS, encoding="utf-8")]
    cap_posts = [p for p in cap_posts if p["tmdb_ids"] and p["post_id"] in row_of]
    logging.info(f"Population: {len(cap_posts)} posts (caption + tmdb_ids).")

    def scored(name, build_vec):
        posts = []
        for p in cap_posts:
            i = row_of[p["post_id"]]
            v = build_vec(p, i)
            if v is None:
                continue
            posts.append({
                "combined_vector": v.tolist(),
                "tmdb_ids": p["tmdb_ids"],
                "descriptors": p["descriptors"],
                "image_exists": bool(hi[i]),
            })
        r = evaluate(posts)
        row = {
            "variant": name,
            "n": len(posts),
            "lift_rare": r["knn"]["lift_rare"],
            "lift_jaccard": r["knn"]["lift_jaccard"],
            "by_mod_rare": {k: v["lift_rare"] for k, v in r["by_modality"].items()},
        }
        logging.info(f"{name}: rare={row['lift_rare']}  by_mod={row['by_mod_rare']}")
        return row

    def siglip_vec(p, i):
        if ht[i] and hi[i]:
            v = SIGLIP_TEXT_W * tv[i] + (1 - SIGLIP_TEXT_W) * iv[i]
        elif hi[i]:
            v = iv[i]
        elif ht[i]:
            v = tv[i]
        else:
            return None  # SigLIP has nothing for these; caption variants do
        return v / np.linalg.norm(v)

    def caption_vec(p, i):
        c = np.asarray(p["combined_vector"], dtype=np.float32)
        return c / np.linalg.norm(c)

    def hybrid_vec(w):
        def f(p, i):
            c = np.asarray(p["combined_vector"], dtype=np.float32)
            c = c / np.linalg.norm(c)
            img = iv[i] if hi[i] else np.zeros_like(c)
            v = np.concatenate([w * c, (1 - w) * img])
            return v / np.linalg.norm(v)
        return f

    rows = [
        scored("A_siglip_w0.1", siglip_vec),
        scored("B_captions_bge", caption_vec),
    ]
    for w in CAPTION_WEIGHTS:
        rows.append(scored(f"C_hybrid_wcap{w}", hybrid_vec(w)))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "hybrid_comparison.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    best = max(rows, key=lambda r: r["lift_rare"])
    print("\n=== THREE-WAY COMPARISON (rare-overlap lift, same population) ===")
    for r in rows:
        marker = "  <-- best" if r is best else ""
        print(f"{r['variant']:<22} rare={r['lift_rare']:>5.2f}  jacc={r['lift_jaccard']:>5.2f}{marker}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
