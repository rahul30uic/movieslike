"""
Two-tower retrieval model for mood -> movie.

Query tower : post hybrid vector (1536)              -> MLP -> 256, L2-norm
Item  tower : [reddit vibe (1536) ; facts (768) ;    -> MLP -> 256, L2-norm
               log(1+support)]
Trained by in-batch sampled-softmax (InfoNCE) on upvote-weighted post->movie
edges, with a logQ popularity correction so the model isn't rewarded for just
predicting popular movies.

Honest split: the item tower's "reddit vibe" input is aggregated from TRAIN
posts only; held-out posts never touch any movie vector. Movies with no train
support get a zero vibe block and rely on facts (the cold-start win). Eval
reuses the held-out post->movie protocol with the support-tier breakdown.

Usage:
    python pipeline/train_two_tower.py --epochs 30
"""

import argparse
import json
import logging
import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
RESULTS_DIR = os.path.join(REPO, "eval", "eval_results")
POSTS_FILE = os.path.join(DATA_DIR, "posts_with_hybrid_vectors.json")
EDGES_FILE = os.path.join(DATA_DIR, "movie_edge_scores.jsonl")
FACTS_FILE = os.path.join(DATA_DIR, "movie_facts_vectors.npz")

SEED = 42
VAL_FRACTION = 0.2
DIM = 256
TEMP = 0.05
BATCH = 512
LR = 1e-3
UPVOTE_FLOOR = 0.05
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TIERS = [("0", 0, 0), ("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9),
         ("10-24", 10, 24), ("25+", 25, 10**9)]


def has_signal(p):
    return (isinstance(p.get("descriptors"), list) and len(p["descriptors"]) > 0) or bool(p.get("image_exists"))


def load(facts_file=FACTS_FILE):
    posts = []
    for line in open(POSTS_FILE, encoding="utf-8"):
        p = json.loads(line)
        if isinstance(p.get("combined_vector"), list) and p.get("tmdb_ids") and has_signal(p):
            posts.append(p)
    agree = {}
    for line in open(EDGES_FILE, encoding="utf-8"):
        e = json.loads(line)
        if e.get("agreement_norm") is not None:
            agree[(e["post_id"], e["tmdb_id"])] = e["agreement_norm"]
    fz = np.load(facts_file)
    facts = {int(t): v for t, v in zip(fz["tmdb_ids"], fz["vectors"].astype(np.float32))}
    logging.info(f"{len(posts)} posts, {len(agree)} scored edges, {len(facts)} facts vectors.")
    return posts, agree, facts


def build(posts, agree, facts, val_fraction=VAL_FRACTION):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(posts))
    n_val = int(len(posts) * val_fraction)
    test = [posts[i] for i in order[:n_val]]
    train = [posts[i] for i in order[n_val:]]

    # movie universe = movies with facts (servable catalog)
    universe = sorted(facts.keys())
    mi = {tid: i for i, tid in enumerate(universe)}

    # reddit vibe input, aggregated from TRAIN posts only (upvote-weighted)
    vibe_sum = np.zeros((len(universe), 1536), dtype=np.float32)
    vibe_w = np.zeros(len(universe), dtype=np.float32)
    support = np.zeros(len(universe), dtype=np.float32)
    for p in train:
        v = np.asarray(p["combined_vector"], dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-9
        for tid in set(p["tmdb_ids"]):
            if tid in mi:
                w = UPVOTE_FLOOR + max(agree.get((p["post_id"], tid), 0.0), 0.0)
                vibe_sum[mi[tid]] += w * v
                vibe_w[mi[tid]] += w
                support[mi[tid]] += 1
    nz = vibe_w > 0
    vibe_sum[nz] /= vibe_w[nz, None]

    facts_mat = np.stack([facts[t] for t in universe]).astype(np.float32)
    log_sup = np.log1p(support).astype(np.float32)[:, None]
    item_feat = np.concatenate([vibe_sum, facts_mat, log_sup], axis=1)  # (U, 2305)

    # training edges (train posts -> movies in universe), agreement-weighted
    q_train = np.stack([unit(np.asarray(p["combined_vector"], dtype=np.float32)) for p in train])
    edges = []  # (train_post_idx, movie_idx, weight)
    freq = Counter()
    for pi, p in enumerate(train):
        for tid in set(p["tmdb_ids"]):
            if tid in mi:
                w = UPVOTE_FLOOR + max(agree.get((p["post_id"], tid), 0.0), 0.0)
                edges.append((pi, mi[tid], w))
                freq[mi[tid]] += 1
    logQ = np.zeros(len(universe), dtype=np.float32)
    tot = sum(freq.values())
    for j, c in freq.items():
        logQ[j] = math.log(c / tot)

    return dict(train=train, test=test, universe=universe, mi=mi, support=support,
                item_feat=item_feat, q_train=q_train, edges=edges, logQ=logQ)


def unit(x):
    return x / (np.linalg.norm(x) + 1e-9)


class Tower(nn.Module):
    def __init__(self, in_dim, out_dim=DIM, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def evaluate(qtower, itower, D):
    qtower.eval(); itower.eval()
    with torch.no_grad():
        I = itower(torch.tensor(D["item_feat"], device=DEVICE))          # (U,256)
        mi, support = D["mi"], D["support"]
        tier_hits = {n: [] for n, _, _ in TIERS}
        rr_all, hit10 = [], []
        Q = qtower(torch.tensor(np.stack([unit(np.asarray(p["combined_vector"], np.float32))
                                          for p in D["test"]]), device=DEVICE))
        sims = Q @ I.T                                                    # (T,U)
        for r, p in enumerate(D["test"]):
            rel = [mi[t] for t in set(p["tmdb_ids"]) if t in mi]
            if not rel:
                continue
            order = torch.argsort(sims[r], descending=True)
            rank_of = torch.empty(len(order), dtype=torch.long, device=DEVICE)
            rank_of[order] = torch.arange(len(order), device=DEVICE)
            ranks = {j: int(rank_of[j]) + 1 for j in rel}
            best = min(ranks.values())
            rr_all.append(1.0 / best); hit10.append(1.0 if best <= 10 else 0.0)
            for j in rel:
                s = support[j]
                tn = next(n for n, lo, hi in TIERS if lo <= s <= hi)
                tier_hits[tn].append(1.0 if ranks[j] <= 10 else 0.0)
    tiers = {n: round(float(np.mean(v)), 4) if v else None for n, v in tier_hits.items()}
    return round(float(np.mean(rr_all)), 4), round(float(np.mean(hit10)), 4), tiers


def export_serving(qtower, itower, D):
    """Write the browser serving artifacts: 256-d item index, movie metadata,
    and the (small) query-tower weights so the browser can map a query vector
    into the learned space."""
    engine_dir = os.path.join(REPO, "frontend", "public", "engine")
    qtower.eval(); itower.eval()
    with torch.no_grad():
        I = itower(torch.tensor(D["item_feat"], device=DEVICE)).cpu().numpy().astype(np.float16)

    meta = {}
    for line in open(os.path.join(DATA_DIR, "movie_vectors_hybrid.json"), encoding="utf-8"):
        m = json.loads(line)
        meta[m["tmdb_id"]] = (m.get("title", ""), m.get("poster_path"), m.get("vote_count") or 0)
    universe, support = D["universe"], D["support"]
    movies = [{"id": int(t), "t": meta.get(t, ("", None, 0))[0],
               "p": meta.get(t, ("", None, 0))[1], "n": int(support[i]),
               "v": int(meta.get(t, ("", None, 0))[2])}
              for i, t in enumerate(universe)]

    I.tofile(os.path.join(engine_dir, "tt_items.bin"))
    json.dump({"dim": I.shape[1], "movies": movies},
              open(os.path.join(engine_dir, "tt_movies.json"), "w"))

    # query tower = Linear(1536,512) -> GELU -> Dropout -> Linear(512,256)
    # weights shipped as one fp16 blob (w1,b1,w2,b2), shapes in the json.
    net = qtower.net
    w = {"w1": net[0].weight.detach().cpu().numpy(), "b1": net[0].bias.detach().cpu().numpy(),
         "w2": net[3].weight.detach().cpu().numpy(), "b2": net[3].bias.detach().cpu().numpy()}
    blob = np.concatenate([w[k].ravel() for k in ("w1", "b1", "w2", "b2")]).astype(np.float16)
    blob.tofile(os.path.join(engine_dir, "tt_query_tower.bin"))
    json.dump({"order": ["w1", "b1", "w2", "b2"],
               "shapes": {k: list(v.shape) for k, v in w.items()}},
              open(os.path.join(engine_dir, "tt_query_tower.json"), "w"))

    size = os.path.getsize(os.path.join(engine_dir, "tt_items.bin")) / 1e6
    qsz = os.path.getsize(os.path.join(engine_dir, "tt_query_tower.json")) / 1e6
    logging.info(f"Exported {len(movies)} item vectors ({I.shape[1]}-d, {size:.1f}MB), "
                 f"query tower ({qsz:.1f}MB) -> {engine_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--tail_beta", type=float, default=0.0,
                    help="Inverse-support exponent on positive weights (0=off).")
    ap.add_argument("--pop_scale", type=float, default=1.0,
                    help="Multiplier on the logQ popularity correction.")
    ap.add_argument("--facts_file", default=FACTS_FILE,
                    help="Item facts vectors (e.g. movie_vibefacts_vectors.npz).")
    ap.add_argument("--vibe_dropout", type=float, default=0.0,
                    help="Prob. of zeroing a positive's reddit-vibe block in training "
                         "(forces the facts pathway to work → cold-start).")
    ap.add_argument("--query_dropout", type=float, default=0.0,
                    help="Prob. of zeroing the query's caption or image half in training "
                         "(robustness to single-modality serving queries).")
    ap.add_argument("--dump", default=None,
                    help="After training, dump held-out query + item vectors here "
                         "(for the reranker eval).")
    ap.add_argument("--production", action="store_true",
                    help="Train on ALL posts (no held-out) and export serving artifacts.")
    args = ap.parse_args()
    torch.manual_seed(SEED)

    posts, agree, facts = load(args.facts_file)
    D = build(posts, agree, facts, val_fraction=0.0 if args.production else VAL_FRACTION)
    logging.info(f"universe={len(D['universe'])}  train_edges={len(D['edges'])}  "
                 f"test={len(D['test'])}")

    qtower = Tower(1536).to(DEVICE)
    itower = Tower(D["item_feat"].shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(list(qtower.parameters()) + list(itower.parameters()),
                            lr=LR, weight_decay=1e-4)

    Qtr = torch.tensor(D["q_train"], device=DEVICE)
    IF = torch.tensor(D["item_feat"], device=DEVICE)
    logQ = torch.tensor(D["logQ"] * args.pop_scale, device=DEVICE)
    edges = np.asarray([(a, b) for a, b, _ in D["edges"]])
    ew_np = np.asarray([w for _, _, w in D["edges"]], dtype=np.float32)
    # tail emphasis: up-weight positives for low-support movies
    if args.tail_beta > 0:
        sup_e = D["support"][edges[:, 1]]
        tail_w = 1.0 / np.power(sup_e + 1.0, args.tail_beta)
        ew_np = ew_np * tail_w
    ew_np *= len(ew_np) / ew_np.sum()  # normalize mean~1 to keep loss scale stable
    ew = torch.tensor(ew_np, device=DEVICE)
    logging.info(f"tail_beta={args.tail_beta}  pop_scale={args.pop_scale}")
    rng = np.random.default_rng(SEED)

    best = {"mrr": 0.0}
    for epoch in range(1, args.epochs + 1):
        qtower.train(); itower.train()
        perm = rng.permutation(len(edges))
        losses = []
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            if len(idx) < 16:
                continue
            pi = edges[idx, 0]; mj = edges[idx, 1]
            qfeat = Qtr[pi]
            if args.query_dropout > 0:                 # robustness to 1-modality queries
                qfeat = qfeat.clone()
                r = torch.rand(len(pi), device=DEVICE)
                qfeat[r < args.query_dropout / 2, 768:] = 0.0          # image-only
                qfeat[(r >= args.query_dropout / 2) & (r < args.query_dropout), :768] = 0.0  # text-only
                qfeat = F.normalize(qfeat, dim=-1)
            q = qtower(qfeat)                          # (B,256)
            feat = IF[mj]
            if args.vibe_dropout > 0:                  # simulate cold-start: facts only
                feat = feat.clone()
                drop = torch.rand(len(mj), device=DEVICE) < args.vibe_dropout
                feat[drop, :1536] = 0.0                # zero reddit-vibe block
                feat[drop, -1] = 0.0                   # and its log-support signal
            it = itower(feat)                          # (B,256) positives = candidates
            logits = (q @ it.T) / TEMP - logQ[mj][None, :]   # logQ popularity correction
            target = torch.arange(len(idx), device=DEVICE)
            loss = (F.cross_entropy(logits, target, reduction="none") * ew[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if (epoch % 3 == 0 or epoch == args.epochs) and D["test"]:
            mrr, hit10, tiers = evaluate(qtower, itower, D)
            logging.info(f"epoch {epoch:2d}  loss {np.mean(losses):.3f}  "
                         f"MRR {mrr}  Hit@10 {hit10}  tail(0/1/2-4)="
                         f"{tiers['0']}/{tiers['1']}/{tiers['2-4']}")
            if mrr > best["mrr"]:
                best = {"mrr": mrr, "hit10": hit10, "tiers": tiers, "epoch": epoch}
        elif not D["test"]:
            logging.info(f"epoch {epoch:2d}  loss {np.mean(losses):.3f}")

    if args.production:
        export_serving(qtower, itower, D)
        return

    print("\n=== TWO-TOWER (held-out post->movie) — best epoch", best["epoch"], "===")
    print("MRR", best["mrr"], " Hit@10", best["hit10"])
    print("recall@10 by support tier:", best["tiers"])
    with open(os.path.join(RESULTS_DIR, "two_tower.json"), "w") as f:
        json.dump(best, f, indent=2)

    if args.dump:
        qtower.eval(); itower.eval()
        with torch.no_grad():
            I = itower(torch.tensor(D["item_feat"], device=DEVICE)).cpu().numpy()
            Qt = qtower(torch.tensor(np.stack(
                [unit(np.asarray(p["combined_vector"], np.float32)) for p in D["test"]]),
                device=DEVICE)).cpu().numpy()
        votes = {}
        for line in open(os.path.join(DATA_DIR, "movie_vectors_hybrid.json"), encoding="utf-8"):
            m = json.loads(line)
            votes[m["tmdb_id"]] = m.get("vote_count") or 0
        universe = D["universe"]
        item_votes = np.asarray([votes.get(t, 0) for t in universe], dtype=np.float32)
        # relevant universe-indices per test post
        mi = D["mi"]
        test_rel = [[mi[t] for t in set(p["tmdb_ids"]) if t in mi] for p in D["test"]]
        np.savez_compressed(args.dump, q_test=Qt.astype(np.float32),
                            item_vecs=I.astype(np.float32),
                            item_support=D["support"].astype(np.float32),
                            item_votes=item_votes,
                            item_tmdb=np.asarray(universe, dtype=np.int64))
        with open(args.dump.replace(".npz", "_rel.json"), "w") as f:
            json.dump(test_rel, f)
        logging.info(f"Dumped eval vectors to {args.dump} "
                     f"(q_test {Qt.shape}, items {I.shape}).")


if __name__ == "__main__":
    main()
