"""
Simulated-user eval for the head-space probe.

Measures the probe's SEARCH STRATEGY (not the embedding — see below): pick a
held-out post as the user's true mood theta*, simulate a user who each round
picks whichever shown image is closer (full-space cosine) to theta* with
Bradley-Terry noise, run the probe, and measure how close the final
recommendation lands.

Because the simulated user judges in FULL space while the probe may reason in
a reduced "dial" space, this directly tests whether a low-dim probe still
satisfies a full-dimensional preference.

Configs (space / pair-selection / output) let us A/B every planned change:
  raw    current 1536-D probe (baseline)
  dialK  probe reasons in the top-K whitened dials

Metrics, averaged over theta* and swept over sim-noise:
  conv_cos   cosine(final target, theta*)   [higher = converged closer]
  movie_mrr  MRR of theta*'s movies ranked by the final target
  rounds     rounds used (fixed 5 here; adaptive later)

Usage:
    python eval/probe_eval.py                 # baseline (raw) + dial-10
    python eval/probe_eval.py --k 10 --n 300
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from movie_retrieval_eval import load_posts  # noqa: E402  (reuse loader)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
ENGINE_DIR = os.path.join(REPO, "frontend", "public", "engine")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")

# match frontend/lib/engine.js
PROBE_SHARPNESS = 25.0
PROBE_ROUNDS = 5
PROBE_CANDIDATES = 24
PICK_DECISIVENESS = 1.4
SEED = 42
N_THETA = 300
NOISE_BETAS = [1e9, 300.0, 100.0]  # perfect -> realistic -> sloppy (sim rationality)


def unit(x):
    return x / (np.linalg.norm(x) + 1e-9)


def load_pool():
    meta = json.load(open(os.path.join(ENGINE_DIR, "probe_posts.json"), encoding="utf-8"))
    raw = np.fromfile(os.path.join(ENGINE_DIR, "probe_posts.bin"), dtype=np.float16)
    V = raw.astype(np.float32).reshape(len(meta["posts"]), meta["dim"])
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    ids = {p["id"] for p in meta["posts"]}
    return ids, V


def load_dial(k):
    d = np.load(os.path.join(DATA_DIR, "dial_space.npz"))
    return d["mean"], d["components"][:k], d["scales"][:k]


def load_movies():
    ids, vecs = [], []
    for line in open(os.path.join(DATA_DIR, "movie_vectors_hybrid.json"), encoding="utf-8"):
        m = json.loads(line)
        ids.append(m["tmdb_id"])
        vecs.append(m["vector"])
    M = np.asarray(vecs, dtype=np.float32)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return ids, M


def load_thetas(pool_ids, n):
    posts = [p for p in load_posts() if p["post_id"] not in pool_ids]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(posts), min(n, len(posts)), replace=False)
    out = []
    for i in idx:
        v = unit(np.asarray(posts[i]["combined_vector"], dtype=np.float32))
        out.append((v, set(posts[i]["tmdb_ids"])))
    return out


class Probe:
    """Port of engine.js probe. `space` picks the reasoning coordinates."""

    def __init__(self, Vfull, space="raw", dial=None):
        self.Vfull = Vfull
        if space == "raw":
            self.P = Vfull
            self.recon = lambda d: d
        else:
            mean, comps, scales = dial
            self.P = ((Vfull - mean) @ comps.T) / scales           # (n, k) whitened
            self.recon = lambda d: mean + (d * scales) @ comps      # dial -> full
        self.G = self.P @ self.P.T                                  # pairwise, cached
        self.n = len(Vfull)

    def posterior(self, history):
        w = np.ones(self.n)
        for chosen, rejected in history:
            margins = self.G[:, chosen] - self.G[:, rejected]
            z = (margins - margins.mean()) / (margins.std() + 1e-9)
            w *= 1.0 / (1.0 + np.exp(-PICK_DECISIVENESS * z))
        s = w.sum()
        return w / s if s > 0 else np.full(self.n, 1.0 / self.n)

    def next_pair(self, history, w, rng):
        shown = {i for h in history for i in h}
        avail = np.array([i for i in range(self.n) if i not in shown])
        wa = w[avail] / w[avail].sum()
        a = int(rng.choice(avail, p=wa))
        simsA = self.G[:, a]
        cands = set()
        for _ in range(PROBE_CANDIDATES * 3):
            if len(cands) >= PROBE_CANDIDATES:
                break
            c = int(rng.choice(avail, p=wa))
            if c != a:
                cands.add(c)
        progress = min(len(history) / (PROBE_ROUNDS - 1), 1.0)
        contrast = 0.8 * (1 - progress) + 0.1
        best, best_score = None, np.inf
        for c in cands:
            z = np.clip(PROBE_SHARPNESS * (simsA - self.G[:, c]), -30, 30)
            pA = float(np.sum(w / (1.0 + np.exp(-z))))
            score = abs(pA - 0.5) + contrast * self.G[a, c]
            if score < best_score:
                best_score, best = score, c
        return a, best

    def target(self, history):
        w = self.posterior(history)
        tp = (self.P * w[:, None]).sum(0)
        return unit(self.recon(tp))


def sim_pick(theta, va, vb, beta, rng):
    m = float(va @ theta - vb @ theta)
    if beta >= 1e8:
        return 0 if m >= 0 else 1
    p = 1.0 / (1.0 + np.exp(-beta * m))
    return 0 if rng.random() < p else 1


def whiten_project(v, wdial):
    """Project a full-space vector into the whitened dial space used only for
    the metric — removes the common-mode anisotropy so convergence is visible."""
    mean, comps, scales = wdial
    return ((v - mean) @ comps.T) / scales


def run(config, Vfull, thetas, movies, dial, wdial):
    ids_m, M = movies
    probe = Probe(Vfull, space=config["space"], dial=dial)
    rng = np.random.default_rng(SEED)
    rows = {b: {"conv": [], "mrr": []} for b in NOISE_BETAS}
    for theta, tmdb in thetas:
        theta_d = whiten_project(theta, wdial)
        rel = [i for i, t in enumerate(ids_m) if t in tmdb]
        for beta in NOISE_BETAS:
            if config["space"] == "none":            # no-probe floor: global centroid
                tgt = unit(Vfull.mean(0))
            else:
                history = []
                for _ in range(PROBE_ROUNDS):
                    w = probe.posterior(history)
                    a, b = probe.next_pair(history, w, rng)
                    pick = sim_pick(theta, Vfull[a], Vfull[b], beta, rng)
                    chosen, rejected = (a, b) if pick == 0 else (b, a)
                    history.append((chosen, rejected))
                tgt = probe.target(history)
            # convergence in the DISCRIMINATIVE (whitened) space
            td = whiten_project(tgt, wdial)
            rows[beta]["conv"].append(float(unit(td) @ unit(theta_d)))
            if rel:
                order = np.argsort(-(M @ tgt))
                ranks = {int(j): r + 1 for r, j in enumerate(order)}
                rows[beta]["mrr"].append(1.0 / min(ranks[i] for i in rel))
    return {b: {"conv_white": round(float(np.mean(rows[b]["conv"])), 4),
                "movie_mrr": round(float(np.mean(rows[b]["mrr"])), 4)}
            for b in NOISE_BETAS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=N_THETA)
    args = ap.parse_args()

    pool_ids, Vfull = load_pool()
    thetas = load_thetas(pool_ids, args.n)
    movies = load_movies()
    dial = load_dial(args.k)
    wdial = load_dial(15)  # fixed 15-dim whitened space for the metric only
    print(f"pool={len(Vfull)}  theta*={len(thetas)}  movies={len(movies[0])}  dial_k={args.k}")

    configs = [
        {"name": "no-probe (floor)", "space": "none"},
        {"name": "raw (baseline)", "space": "raw"},
        {"name": f"dial-{args.k} (whitened)", "space": "dial"},
    ]
    results = {}
    for cfg in configs:
        results[cfg["name"]] = run(cfg, Vfull, thetas, movies, dial, wdial)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "probe_eval.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== PROBE EVAL (sim user, 5 rounds) — conv_cos / movie_mrr by sim noise ===")
    print(f"{'config':<22}{'perfect':>16}{'realistic':>16}{'sloppy':>16}")
    for name, r in results.items():
        cells = "".join(f"{r[b]['conv_white']:.3f}/{r[b]['movie_mrr']:.3f}".rjust(16)
                        for b in NOISE_BETAS)
        print(f"{name:<22}{cells}")
    print("\n(higher is better; left number = cosine to true mood, right = movie MRR)")


if __name__ == "__main__":
    main()
