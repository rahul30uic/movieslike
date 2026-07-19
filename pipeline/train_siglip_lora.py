"""
LoRA fine-tune of SigLIP's vision tower on the corpus's own supervision:
two posts whose commenters recommended the same RARE movie share a vibe, so
their IMAGES are a positive pair for image-image contrastive learning.

Protocol mirrors train_projection_head.py: posts split 80/20 before pair
building; the score is image-block-only rare-overlap lift@10 within the val
pool, tuned tower vs frozen tower on the identical population.

Usage:
    python pipeline/train_siglip_lora.py --epochs 6
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoImageProcessor, SiglipVisionModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
sys.path.insert(0, os.path.join(REPO_ROOT, "eval"))
from eval_embeddings import evaluate, HEAD_QUANTILE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

POSTS_FILE = os.path.join(DATA_DIR, "posts_with_hybrid_vectors.json")
DATASET_CSV = os.path.join(DATA_DIR, "final_dataset.csv")
MODEL_NAME = "google/siglip-base-patch16-224"

SEED = 42
VAL_FRACTION = 0.2
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 24
LR = 1e-4
TEMP = 0.07


def load_posts_with_images():
    paths = {}
    for _, row in pd.read_csv(DATASET_CSV).iterrows():
        p = row.get("image_local_path")
        if isinstance(p, str) and p.strip():
            first = p.split("|")[0].strip()
            ap = os.path.join(DATA_DIR, first)
            if os.path.exists(ap):
                paths[row["post_id"]] = ap
    posts = []
    for line in open(POSTS_FILE, encoding="utf-8"):
        p = json.loads(line)
        if p["post_id"] in paths and p.get("image_exists") and p.get("tmdb_ids"):
            posts.append({"post_id": p["post_id"], "path": paths[p["post_id"]],
                          "tmdb_ids": p["tmdb_ids"], "descriptors": p.get("descriptors", [])})
    logging.info(f"{len(posts)} posts with images + movies.")
    return posts


def rare_sets(posts):
    freq = Counter(m for p in posts for m in set(p["tmdb_ids"]))
    counts = np.array(sorted(freq.values()))
    cutoff = counts[int(len(counts) * HEAD_QUANTILE)]
    head = {m for m, c in freq.items() if c >= cutoff}
    return [set(p["tmdb_ids"]) - head for p in posts]


def build_pairs(indices, rs):
    by_movie = defaultdict(list)
    for i in indices:
        for m in rs[i]:
            by_movie[m].append(i)
    partners = defaultdict(set)
    for m, members in by_movie.items():
        if 2 <= len(members) <= 30:
            for i in members:
                partners[i].update(j for j in members if j != i)
    return {i: list(js) for i, js in partners.items() if js}


@torch.no_grad()
def encode_all(model, processor, posts, indices):
    model.eval()
    vecs = np.zeros((len(indices), 768), dtype=np.float32)
    for s in range(0, len(indices), 64):
        chunk = indices[s:s + 64]
        images = [Image.open(posts[i]["path"]).convert("RGB") for i in chunk]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        out = model(**inputs).pooler_output
        out = torch.nn.functional.normalize(out, dim=-1)
        vecs[s:s + len(chunk)] = out.float().cpu().numpy()
    return vecs


def val_lift(vecs, posts, val_idx, val_rows):
    pool = [{"combined_vector": vecs[val_rows[i]].tolist(), "tmdb_ids": posts[i]["tmdb_ids"],
             "descriptors": posts[i]["descriptors"], "image_exists": True} for i in val_idx]
    return evaluate(pool)["knn"]["lift_rare"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    posts = load_posts_with_images()
    rs = rare_sets(posts)
    order = rng.permutation(len(posts))
    n_val = int(len(posts) * VAL_FRACTION)
    val_idx, train_idx = list(order[:n_val]), list(order[n_val:])
    pairs = build_pairs(train_idx, rs)
    anchors = list(pairs)
    logging.info(f"Train {len(train_idx)} / val {len(val_idx)}; {len(anchors)} anchors.")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SiglipVisionModel.from_pretrained(MODEL_NAME).to(DEVICE)

    val_rows = {i: k for k, i in enumerate(val_idx)}
    base_vecs = encode_all(model, processor, posts, val_idx)
    base_vecs_map = np.zeros((len(posts), 768), dtype=np.float32)
    for i, k in val_rows.items():
        base_vecs_map[k] = base_vecs[k]
    baseline = val_lift(base_vecs_map, posts, val_idx, val_rows)
    logging.info(f"BASELINE (frozen tower) val image-only rare-lift: {baseline}")

    lora = LoraConfig(r=args.rank, lora_alpha=16, lora_dropout=0.1,
                      target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    best = {"epoch": 0, "lift": baseline}
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(anchors)
        losses = []
        for s in range(0, len(anchors), BATCH):
            batch = anchors[s:s + BATCH]
            if len(batch) < 8:
                continue
            pos = [int(rng.choice(pairs[a])) for a in batch]
            images = [Image.open(posts[i]["path"]).convert("RGB") for i in batch + pos]
            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            out = model(**inputs).pooler_output
            out = torch.nn.functional.normalize(out, dim=-1)
            za, zp = out[:len(batch)], out[len(batch):]
            logits = za @ zp.T / TEMP
            labels = torch.arange(len(batch), device=DEVICE)
            loss = (torch.nn.functional.cross_entropy(logits, labels)
                    + torch.nn.functional.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        vecs = encode_all(model, processor, posts, val_idx)
        vmap = np.zeros((len(posts), 768), dtype=np.float32)
        for i, k in val_rows.items():
            vmap[k] = vecs[k]
        lift = val_lift(vmap, posts, val_idx, val_rows)
        logging.info(f"epoch {epoch}: loss {np.mean(losses):.3f}  val rare-lift {lift}")
        if lift > best["lift"]:
            best = {"epoch": epoch, "lift": lift}
            model.save_pretrained(os.path.join(DATA_DIR, "siglip_lora_best"))

    result = {"baseline_val_rare_lift": baseline, "best": best,
              "gain_pct": round(100 * (best["lift"] - baseline) / baseline, 1),
              "rank": args.rank, "epochs": args.epochs,
              "n_train": len(train_idx), "n_val": len(val_idx)}
    out = os.path.join(REPO_ROOT, "eval", "eval_results", "siglip_lora.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
