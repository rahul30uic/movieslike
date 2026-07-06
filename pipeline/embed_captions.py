"""
Embed VLM vibe captions with a strong text-only embedding model and emit a
posts file in the same schema eval_embeddings.py consumes.

This is the second half of the caption approach: Gemini fuses image + title +
descriptors into one paragraph (generate_vibe_captions.py), and a proper
sentence-embedding model — not SigLIP's weak text tower — turns it into the
post's vector. One text-native space for posts, movies, and (later) user chat
queries.

Usage:
    python embed_captions.py
    python embed_captions.py --eval          # also score with the eval harness
"""

import argparse
import ast
import json
import logging
import os

import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
CAPTIONS_FILE = os.path.join(DATA_DIR, "vibe_captions.jsonl")
INPUT_CSV = os.path.join(DATA_DIR, "final_dataset.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "posts_with_caption_vectors.json")

MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEVICE = "mps"
BATCH_SIZE = 64


def safe_list(val):
    if pd.isna(val):
        return []
    try:
        v = ast.literal_eval(val)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="Run eval harness on the output.")
    args = ap.parse_args()

    captions = {}
    with open(CAPTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                captions[r["post_id"]] = r["caption"]
            except (json.JSONDecodeError, KeyError):
                pass
    logging.info(f"Loaded {len(captions)} captions.")

    df = pd.read_csv(INPUT_CSV)
    df["descriptors"] = df["descriptors"].apply(safe_list)
    df["tmdb_ids"] = df["tmdb_ids"].apply(safe_list)
    df["has_image"] = df["image_local_path"].notna()
    df = df[df["post_id"].isin(captions)].reset_index(drop=True)
    logging.info(f"{len(df)} posts have captions.")

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    texts = [captions[pid] for pid in df["post_id"]]
    vecs = model.encode(texts, batch_size=BATCH_SIZE, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            f.write(json.dumps({
                "post_id": row["post_id"],
                "descriptors": row["descriptors"],
                "tmdb_ids": row["tmdb_ids"],
                "image_exists": bool(row["has_image"]),
                "caption": captions[row["post_id"]],
                "combined_vector": vecs[i].tolist(),
            }) + "\n")
    logging.info(f"Wrote {len(df)} posts to {OUTPUT_FILE}.")

    if args.eval:
        import sys; sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), "eval"))
from eval_embeddings import evaluate
        posts = [json.loads(l) for l in open(OUTPUT_FILE, encoding="utf-8")]
        posts = [p for p in posts if p["tmdb_ids"]]
        results = evaluate(posts)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
