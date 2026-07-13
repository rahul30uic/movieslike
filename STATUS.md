# Project Status

> Glanceable state of Movieslike. Update after every working session.
> Vision: [VISION.md](VISION.md) — 5 movies, react don't browse, converge to one.

**Last updated:** 2026-07-06

## Where we are

| Area | State |
|---|---|
| Reddit corpus → verified movies pipeline | ✅ Done (3,730 posts) |
| Post embeddings (SigLIP, text+image avg) | ✅ Done, quality unmeasured |
| Anchor-grid demo (FastAPI + Next.js) | ✅ Working |
| Eval harness (movie-overlap metric) | ✅ Done — `movie extraction/eval_embeddings.py` |
| Movie-level vibe vectors | ✅ Done — `movie extraction/build_movie_vectors.py` → `movie_vectors.json` |
| Fusion-weight fix (quick win) | ✅ Measured — w_text=0.1 beats 0.5 by 29% |
| VLM captions | ✅ 3,715/3,730 (15 safety-filtered) |
| **Embedding verdict: hybrid architecture** | ✅ `[0.5·bge(caption) ; 0.5·SigLIP(image)]` — full-corpus rare lift **7.51** (was 5.01 at baseline, +50%) |
| Movie vectors in hybrid space | ✅ `movie_vectors_hybrid.json` (17k movies) |
| Retrieval API v2 (filters + negatives + text query) | ⬜ Planned |
| Chat MVP (LLM orchestrator + 5-card UI) | ⬜ Planned — this is the MVP |
| Image probe rounds (adaptive) | ⬜ Planned |
| Commit flow (availability, seen-list, follow-up) | ⬜ Planned |

## Key numbers

| Metric | Value | Notes |
|---|---|---|
| Posts in final dataset | 3,730 | |
| …with images | 2,905 (78%) | |
| …with descriptors | 1,863 (50%) | only ~half have text |
| Unique verified movies | 17,531 | 17,174 got vibe vectors |
| …with support ≥ 5 posts | 4,792 (28%) | the confidently-recommendable core |
| …with support = 1 post | 7,725 (45%) | thin tail, use cautiously |
| Baseline eval: overall lift vs random | 2.3× any-overlap, 5.0× rare-overlap | k=10, `eval_results/baseline_siglip_avg.json` |
| …image-only posts | **8.2× rare lift** | images carry the real vibe signal |
| …text+image posts | 4.6× | 50/50 text averaging *dilutes* image signal |
| …text-only posts | **0.2× — below random** | SigLIP text tower is broken for this |
| …no-signal posts (417) | 0.2× — garbage | excluded from movie vectors |

## Decisions log

- **2026-07-04** — Product direction locked: anti-scroll picker (see VISION.md).
  Anchor grid is a stepping stone, not the product.
- **2026-07-04** — Unit of retrieval will be the *movie*, not the post.
  Movie vectors = weighted aggregate of post vectors.
- **2026-07-04** — No embedding changes before the eval harness exists.
- **2026-07-04** — Eval verdict: image embeddings are the gold (8.2× lift);
  SigLIP text embeddings score *below random* and the 50/50 average dilutes
  the image signal. Embedding upgrade priority = fix the text side (VLM
  caption → text embedder, or image-weighted fusion). Every future embedding
  version must beat `eval_results/baseline_siglip_avg.json`.
- **2026-07-04** — Movie vectors: post weight = 1/log2(1+n_recs) to down-weight
  shotgun threads; no-signal posts excluded. Neighbor spot-checks are strong
  (Lost in Translation → Chungking Express/Her; The Thing → cosmic horror).
- **2026-07-04** — Fusion sweep (`eval_results/fusion_sweep.json`): best
  text weight = **0.1** (rare lift 6.56) vs current 0.5 (5.09) — +29% free.
  Text-only posts stay broken (~0.26) at every weight; only the caption
  approach can rescue them. Gemini captions on 50-post sample look excellent;
  full run needs a paid Gemini key (~$2-5) — free tier is 5 req/min.
- **2026-07-06** — Caption sample: 35 captions done (26 flash + 9 flash-lite;
  quality equivalent). Free-tier daily caps are tiny (flash: 20/day) — the
  full 3,730-post run is not feasible without a paid key.
- **2026-07-06** — Full caption run: 3,132/3,730 before prepaid credits ran
  out (2.5-flash "thinking" tokens inflated cost; now disabled via
  thinking_budget=0 — remaining ~600 posts ≈ under $1 after top-up).
- **2026-07-06** — **Embedding verdict** (`eval_results/hybrid_comparison.json`,
  three-way on identical population): captions alone rescue text-only posts
  (0.26→2.29 lift) but lose image nuance (5.5 vs ~7 on image posts). The
  winner is the CONCAT HYBRID `[0.5·caption_bge ; 0.5·siglip_image]`:
  rare lift 6.77 (+15% over best SigLIP fusion). Cosine on the concat =
  weighted sum of per-space cosines, so no cross-space alignment needed, and
  text chat queries can search the caption block directly. Known wart:
  text-only posts have a zero image block, which reintroduces a small gap
  for them (0.88) — acceptable at 8% of corpus; revisit via image-block
  imputation if it matters.

- **2026-07-13** — Trained component shipped: residual contrastive projection
  head (zero-init MLP delta over frozen features), positives = rare-movie
  co-recommendation, honest 80/20 post split. Val rare-lift: raw 4.27 →
  trained **4.54 (+6.3%)**. Two instructive failures kept in the log: a
  non-residual head *destroyed* the pretrained geometry (−9%), and raw
  DINOv2 underperformed SigLIP for mood retrieval (3.07 vs 4.27; trained it
  recovers to only 4.16). SigLIP+DINO combined+head: 4.56 — tie, DINO adds
  nothing here. Verdict: caption+SigLIP+head is production. DINOv3 rerun
  pending HF license acceptance (`pipeline/encode_dino.py --model ...`).

## Known issues / risks

- Posts with no descriptors AND no image get the embedding of `""` (garbage
  vectors in the index) — `generate_embeddings_and_clusters.py:196`.
- Modality gap: text-only vs image-only posts may cluster by modality, not vibe.
- Corpus likely skews atmospheric/moody — coverage audit will quantify.
- Duplicate frontend copies (`movie-frontend/` vs `movie extraction/frontend/`);
  `movie-frontend/` is canonical.
- Nearly everything is untracked in git; `.env` with API keys sits inside
  `movie extraction/` — needs `.gitignore` before any commit.
- TMDB verification has mismatches: "The Boys from Brazil: Rise of the
  Bolsonaros" (a docuseries) has 218 posts — almost certainly wrong-title
  matches for the 1978 film. The verification step needs a pass.
- Movies with support = 1 post (45%) have noisy vectors; consider a support
  floor or shrinkage toward a genre prior when recommending.
