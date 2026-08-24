"""
Reranker eval — Stage 2 of the two-stage pipeline.

Stage 1 (two-tower) retrieves the top-K candidates by pure relevance
(dumped by train_two_tower.py --dump). This sweeps the reranking POLICY over
those candidates and reports the multi-objective tradeoff, because no single
number captures a good rerank:

  Recall@10   relevance retention — did we keep the right movies near the top
  diversity   1 - mean pairwise cosine of the top-N (higher = less same-y)
  tail_frac   fraction of the top-N that are long-tail (support < 10)
  mean_pop    mean log(1+support) of the top-N (lower = less head-biased)

Policy: base = z(cosine) - alpha * z(log popularity), min-max normalized over
the K candidates; then MMR picks N that maximize (1-lam)*base - lam*max_sim
to the already-picked (diversity). alpha=lam=0 reduces to plain cosine top-N.

Usage:
    python eval/reranker_eval.py
"""

import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")

K = 200      # candidates from retrieval
N = 12       # final list size
TAIL_SUPPORT = 10


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def mmr(cand, base, vecs, n, lam):
    """Greedy MMR: pick n maximizing (1-lam)*base - lam*max cos to selected."""
    remaining = list(range(len(cand)))
    selected = []
    sim = vecs @ vecs.T  # (K,K) among candidates
    while remaining and len(selected) < n:
        best, best_val = None, -1e9
        for r in remaining:
            div = max((sim[r, s] for s in selected), default=0.0)
            val = (1 - lam) * base[r] - lam * div
            if val > best_val:
                best_val, best = val, r
        selected.append(best)
        remaining.remove(best)
    return [cand[s] for s in selected], selected


def run(cfg, D):
    q, I = D["q_test"], D["item_vecs"]
    support, rels = D["item_support"], D["rel"]
    recall10, div, tailf, pop = [], [], [], []
    for i in range(len(q)):
        rel = set(rels[i])
        if not rel:
            continue
        sims = I @ q[i]
        cand = np.argpartition(sims, -K)[-K:]
        cand = cand[np.argsort(-sims[cand])]            # top-K by cosine
        base = z(sims[cand]) - cfg["alpha"] * z(np.log1p(support[cand]))
        base = (base - base.min()) / (np.ptp(base) + 1e-9)  # -> [0,1] for MMR
        cvecs = I[cand]
        chosen, sel_local = mmr(cand, base, cvecs, N, cfg["lam"])
        top10 = set(chosen[:10])
        recall10.append(len(rel & top10) / len(rel))
        cv = I[chosen]
        pw = cv @ cv.T
        m = ~np.eye(len(chosen), dtype=bool)
        div.append(1.0 - float(pw[m].mean()))
        tailf.append(float(np.mean([support[c] < TAIL_SUPPORT for c in chosen])))
        pop.append(float(np.mean(np.log1p(support[chosen]))))
    return {"Recall@10": round(np.mean(recall10), 4), "diversity": round(np.mean(div), 4),
            "tail_frac": round(np.mean(tailf), 4), "mean_pop": round(np.mean(pop), 3)}


def main():
    d = np.load(os.path.join(DATA_DIR, "two_tower_eval.npz"))
    rel = json.load(open(os.path.join(DATA_DIR, "two_tower_eval_rel.json")))
    D = {"q_test": d["q_test"], "item_vecs": d["item_vecs"],
         "item_support": d["item_support"], "rel": rel}

    configs = [
        ("cosine (no rerank)", {"alpha": 0.0, "lam": 0.0}),
        ("popularity a=0.3", {"alpha": 0.3, "lam": 0.0}),
        ("mmr lam=0.3", {"alpha": 0.0, "lam": 0.3}),
        ("mmr lam=0.5", {"alpha": 0.0, "lam": 0.5}),
        ("pop+mmr a=0.3 lam=0.3", {"alpha": 0.3, "lam": 0.3}),
    ]
    results = {name: run(cfg, D) for name, cfg in configs}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(results, open(os.path.join(RESULTS_DIR, "reranker_eval.json"), "w"), indent=2)

    print(f"\n=== RERANKER EVAL (K={K} candidates -> N={N}) ===")
    print(f"{'policy':<24}{'Recall@10':>11}{'diversity':>11}{'tail_frac':>11}{'mean_pop':>10}")
    for name, r in results.items():
        print(f"{name:<24}{r['Recall@10']:>11}{r['diversity']:>11}{r['tail_frac']:>11}{r['mean_pop']:>10}")
    print("\nRecall@10=relevance kept; diversity/tail_frac higher=better; mean_pop lower=less head-biased")


if __name__ == "__main__":
    main()
