"""
Sweep text/image fusion weights over the separately-encoded modality vectors
and score each variant with the eval harness.

For text weight w: combined = normalize(w * text_vec + (1-w) * image_vec).
Posts with a single modality use that modality's vector regardless of w.
Posts with neither modality are EXCLUDED (garbage embeddings), so all
variants here — including w=0.5, which reproduces the current pipeline's
fusion — are scored on the same 'has-signal' population and are directly
comparable to each other.

Usage:
    python sweep_fusion.py
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

TEXT_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def main():
    data = np.load(NPZ_FILE)
    tv, iv = data["text_vecs"], data["image_vecs"]
    ht, hi = data["has_text"], data["has_image"]
    meta = [json.loads(line) for line in open(META_FILE, encoding="utf-8")]
    assert len(meta) == len(tv)

    # Same population for every variant: has some signal AND has ground truth.
    keep = [i for i in range(len(meta))
            if (ht[i] or hi[i]) and meta[i]["tmdb_ids"]]
    logging.info(f"Evaluating on {len(keep)} posts (signal + >=1 tmdb_id).")

    summary = []
    for w in TEXT_WEIGHTS:
        posts = []
        for i in keep:
            if ht[i] and hi[i]:
                v = w * tv[i] + (1 - w) * iv[i]
                v = v / np.linalg.norm(v)
            elif hi[i]:
                v = iv[i]
            else:
                v = tv[i]
            posts.append({
                "combined_vector": v.tolist(),
                "tmdb_ids": meta[i]["tmdb_ids"],
                "descriptors": meta[i]["descriptors"],
                "image_exists": bool(hi[i]),
            })
        results = evaluate(posts)
        row = {
            "text_weight": w,
            "lift_any": results["knn"]["lift_any"],
            "lift_jaccard": results["knn"]["lift_jaccard"],
            "lift_rare": results["knn"]["lift_rare"],
            "by_modality": {r: v["lift_rare"] for r, v in results["by_modality"].items()},
        }
        summary.append(row)
        logging.info(f"w_text={w:.1f}  lift_rare={row['lift_rare']}  "
                     f"lift_jaccard={row['lift_jaccard']}  by_mod={row['by_modality']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "fusion_sweep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    best = max(summary, key=lambda r: r["lift_rare"])
    print("\n=== FUSION SWEEP (rare-overlap lift vs random, k=10) ===")
    print(f"{'w_text':>7} {'lift_rare':>10} {'lift_jacc':>10} {'lift_any':>9}")
    for r in summary:
        marker = "  <-- best" if r is best else ""
        print(f"{r['text_weight']:>7.1f} {r['lift_rare']:>10.2f} "
              f"{r['lift_jaccard']:>10.2f} {r['lift_any']:>9.2f}{marker}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
