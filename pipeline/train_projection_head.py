"""
Train a contrastive projection head on top of frozen post features.

The corpus supervises its own metric learning: two posts whose commenters
recommended the same RARE movie (outside the popularity head) almost
certainly share a vibe. Those pairs are the positives; in-batch posts are
the negatives (InfoNCE).

  input   [bge(caption) ; image_vec]   (frozen, 1536-dim)
  head    Linear -> GELU -> Linear -> L2-normalize   (512-dim out)
  loss    symmetric InfoNCE over anchor/positive pairs

Honest evaluation: posts are split 80/20 BEFORE pair building; the val score
is rare-overlap lift@10 computed only within the val pool, compared against
the raw (untrained) features on the exact same pool.

Usage:
    python pipeline/train_projection_head.py                          # SigLIP image block
    python pipeline/train_projection_head.py --image-npz data/post_dinov2_base_vectors.npz
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
sys.path.insert(0, os.path.join(REPO_ROOT, "eval"))
from eval_embeddings import evaluate, HEAD_QUANTILE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CAPTION_VECS = os.path.join(DATA_DIR, "posts_with_caption_vectors.json")
META_FILE = os.path.join(DATA_DIR, "post_modality_meta.json")
DEFAULT_IMAGE_NPZ = os.path.join(DATA_DIR, "post_modality_vectors.npz")

SEED = 42
VAL_FRACTION = 0.2
OUT_DIM = 512
HIDDEN = 1024
TEMP = 0.05
BATCH = 256
EPOCHS = 40
LR = 1e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class Head(nn.Module):
    """Residual head: output = normalize(x + mlp(x)), with the mlp's last
    layer zero-initialized. Epoch 0 is exactly the raw features — training
    can only learn a correction, never has to rebuild the pretrained
    geometry from scratch (which 3k anchors can't support)."""

    def __init__(self, in_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, in_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return nn.functional.normalize(x + self.net(x), dim=-1)


def load_features(image_npz):
    posts = [json.loads(l) for l in open(CAPTION_VECS, encoding="utf-8")]
    meta = [json.loads(l) for l in open(META_FILE, encoding="utf-8")]
    row_of = {m["post_id"]: i for i, m in enumerate(meta)}
    img = np.load(image_npz)
    iv, hi = img["image_vecs"].astype(np.float32), img["has_image"]

    feats, keep = [], []
    for p in posts:
        i = row_of.get(p["post_id"])
        if i is None:
            continue
        c = np.asarray(p["combined_vector"], dtype=np.float32)
        c /= np.linalg.norm(c) + 1e-9
        im = iv[i] if hi[i] else np.zeros(iv.shape[1], dtype=np.float32)
        feats.append(np.concatenate([c, im]))
        keep.append(p)
    X = np.stack(feats)
    logging.info(f"Features: {X.shape} from {image_npz.split('/')[-1]}")
    return X, keep


def rare_movie_sets(posts):
    freq = Counter(m for p in posts for m in set(p["tmdb_ids"]))
    counts = np.array(sorted(freq.values()))
    cutoff = counts[int(len(counts) * HEAD_QUANTILE)]
    head = {m for m, c in freq.items() if c >= cutoff}
    return [set(p["tmdb_ids"]) - head for p in posts]


def build_pairs(indices, rare_sets):
    by_movie = defaultdict(list)
    for i in indices:
        for m in rare_sets[i]:
            by_movie[m].append(i)
    partners = defaultdict(set)
    for m, members in by_movie.items():
        if len(members) < 2 or len(members) > 30:  # ultra-shared movies = weak signal
            continue
        for i in members:
            partners[i].update(j for j in members if j != i)
    return {i: list(js) for i, js in partners.items() if js}


def val_score(vectors, posts, indices, name):
    pool = [{**{k: posts[i][k] for k in ("tmdb_ids", "descriptors", "image_exists")},
             "combined_vector": vectors[i].tolist()} for i in indices]
    pool = [p for p in pool if p["tmdb_ids"]]
    r = evaluate(pool)
    lift = r["knn"]["lift_rare"]
    logging.info(f"  val rare-lift [{name}]: {lift}")
    return lift, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-npz", default=DEFAULT_IMAGE_NPZ)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--tag", default=None, help="Label for results/weights files.")
    args = ap.parse_args()
    tag = args.tag or os.path.basename(args.image_npz).replace("post_", "").replace("_vectors.npz", "")

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    X, posts = load_features(args.image_npz)
    rare_sets = rare_movie_sets(posts)

    order = rng.permutation(len(posts))
    n_val = int(len(posts) * VAL_FRACTION)
    val_idx, train_idx = order[:n_val], order[n_val:]
    train_pairs = build_pairs(train_idx, rare_sets)
    anchors = list(train_pairs)
    logging.info(f"Train {len(train_idx)} / val {len(val_idx)} posts; "
                 f"{len(anchors)} anchors with >=1 rare-overlap partner.")

    baseline_lift, _ = val_score(X, posts, val_idx, "raw features")

    Xt = torch.tensor(X, device=DEVICE)
    head = Head(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    best_lift, best_state = -1.0, None
    for epoch in range(args.epochs):
        head.train()
        rng.shuffle(anchors)
        losses = []
        for s in range(0, len(anchors), BATCH):
            batch = anchors[s:s + BATCH]
            if len(batch) < 8:
                continue
            pos = [int(rng.choice(train_pairs[a])) for a in batch]
            za = head(Xt[batch])
            zp = head(Xt[pos])
            logits = za @ zp.T / args.temp
            labels = torch.arange(len(batch), device=DEVICE)
            loss = (nn.functional.cross_entropy(logits, labels)
                    + nn.functional.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        head.eval()
        with torch.no_grad():
            Z = head(Xt).cpu().numpy()
        lift, _ = val_score(Z, posts, val_idx, f"epoch {epoch + 1} (loss {np.mean(losses):.3f})")
        if lift > best_lift:
            best_lift = lift
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    out_w = os.path.join(DATA_DIR, f"projection_head_{tag}.pt")
    torch.save({"state_dict": best_state, "in_dim": X.shape[1],
                "hidden": HIDDEN, "residual": True, "image_npz": args.image_npz}, out_w)
    result = {
        "tag": tag,
        "baseline_val_rare_lift": baseline_lift,
        "trained_val_rare_lift": best_lift,
        "gain_pct": round(100 * (best_lift - baseline_lift) / baseline_lift, 1),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_anchor_posts": len(anchors),
    }
    res_path = os.path.join(REPO_ROOT, "eval", "eval_results", f"projection_head_{tag}.json")
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    logging.info(f"Best head saved to {out_w}")


if __name__ == "__main__":
    main()
