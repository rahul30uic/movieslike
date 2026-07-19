"""
Generate a rights-clean synthetic probe pool with a diffusion model.

Takes the vibe captions at high-information points of the embedding space
(farthest-point sample of the curated probe pool), renders each as a
cinematic still with SD-Turbo, and re-embeds the results into the hybrid
space ([bge(caption) ; siglip(synthetic image)]). Output mirrors the corpus
probe pool's schema, so the engine can swap pools via NEXT_PUBLIC_PROBE_POOL.

Usage:
    python pipeline/generate_diffusion_probes.py --count 96
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_DIR = os.path.join(REPO_ROOT, "frontend", "public", "engine")
IMG_OUT = os.path.join(REPO_ROOT, "frontend", "public", "probe_synth")

MODEL = "stabilityai/sd-turbo"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
STEPS = 2
STYLE = "cinematic film still, anamorphic, atmospheric, no text. "


def farthest_point_sample(vectors, k, seed=7):
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(vectors)))]
    min_dist = 1 - vectors @ vectors[chosen[0]]
    for _ in range(k - 1):
        nxt = int(np.argmax(min_dist))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, 1 - vectors @ vectors[nxt])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=96)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(ENGINE_DIR, "probe_posts.json"), encoding="utf-8"))
    vecs = np.fromfile(os.path.join(ENGINE_DIR, "probe_posts.bin"), dtype=np.float16)
    vecs = vecs.astype(np.float32).reshape(len(meta["posts"]), meta["dim"])
    idx = farthest_point_sample(vecs, min(args.count, len(meta["posts"])))
    captions = [(meta["posts"][i].get("c") or "").rstrip("…").strip() for i in idx]
    captions = [c for c in captions if len(c) > 20]
    logging.info(f"Generating {len(captions)} synthetic probes on {DEVICE}...")

    pipe = AutoPipelineForText2Image.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    os.makedirs(IMG_OUT, exist_ok=True)
    files = []
    for k, cap in enumerate(captions):
        img = pipe(prompt=STYLE + cap, num_inference_steps=STEPS, guidance_scale=0.0,
                   height=512, width=512).images[0]
        fname = f"synth_{k:03d}.jpg"
        img.save(os.path.join(IMG_OUT, fname), "JPEG", quality=80, optimize=True)
        files.append((fname, cap))
        if (k + 1) % 16 == 0:
            logging.info(f"{k + 1}/{len(captions)} generated")

    # Re-embed into the hybrid space
    logging.info("Embedding synthetic probes (bge captions + SigLIP images)...")
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEVICE)
    sig = SentenceTransformer("google/siglip-base-patch16-224", device=DEVICE)

    cvecs = bge.encode([c for _, c in files], normalize_embeddings=True)
    images = [Image.open(os.path.join(IMG_OUT, f)).convert("RGB") for f, _ in files]
    ivecs = sig.encode(images, batch_size=32, convert_to_numpy=True)
    ivecs = ivecs / (np.linalg.norm(ivecs, axis=1, keepdims=True) + 1e-9)

    hybrid = np.concatenate([0.5 * cvecs, 0.5 * ivecs], axis=1)
    hybrid = hybrid / (np.linalg.norm(hybrid, axis=1, keepdims=True) + 1e-9)
    hybrid.astype(np.float16).tofile(os.path.join(ENGINE_DIR, "probe_synth.bin"))

    with open(os.path.join(ENGINE_DIR, "probe_synth.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dim": int(hybrid.shape[1]),
            "base": "/probe_synth/",
            "posts": [{"id": f_, "f": f_, "c": c} for f_, c in files],
        }, f)

    total = sum(os.path.getsize(os.path.join(IMG_OUT, f)) for f, _ in files)
    logging.info(f"Done: {len(files)} synthetic probes ({total / 1e6:.1f}MB) + hybrid vectors.")


if __name__ == "__main__":
    main()
