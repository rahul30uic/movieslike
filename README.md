# Movieslike

**A vibe-based movie discovery engine, built from 4,400 Reddit posts where people
share an image or a mood and ask: "what movies feel like this?"**

Streaming services cause choice paralysis — endless scrolling, and even after
picking something, the nagging sense that something better is one scroll away.
Movieslike inverts the interaction: instead of a wall of options, it converges
you to a handful of movies that match your *head-space* — expressed in words,
or, where words fail, by reacting to mood images. Full product thesis in
[VISION.md](VISION.md); build state in [STATUS.md](STATUS.md).

This repo is also an end-to-end ML case study: **a measurement-first embedding
investigation that found the initial multimodal fusion was actively destroying
signal, and fixed it for a +50% retrieval improvement** — every claim below is
reproducible from a script in this repo.

---

## The interesting part: evaluation without labels

"Do these two Reddit posts express the same vibe?" has no ground-truth labels —
but the corpus contains a signal nobody has to annotate: **each post's comment
section is a crowd-validated answer key** (the movies people recommended,
verified against TMDB). If two posts' commenters recommend overlapping movies,
the posts almost certainly share a vibe.

The eval ([eval/eval_embeddings.py](eval/eval_embeddings.py)): for every post,
retrieve its 10 nearest neighbors in embedding space and measure movie-overlap
against a random-pairs baseline. The headline metric is **rare-overlap lift** —
overlap counting only movies *outside* the popularity head, because surfacing
the long tail is the whole point (every vibe query would otherwise collapse to
Blade Runner).

### What the eval revealed

The initial architecture embedded descriptor tags and images with SigLIP and
averaged them 50/50. The eval showed this was hurting:

| Post type | Rare-overlap lift vs random (baseline) |
|---|---|
| image-only | **8.2×** — images carry real vibe signal |
| text + image | 4.6× — *worse than image-only: the text average dilutes it* |
| text-only | **0.2× — below random** |

SigLIP's text tower, trained on alt-text captions, embeds descriptor bags like
`"gritty lonely hopeless neo futurism"` somewhere useless — and the classic
CLIP *modality gap* means text-only and image-only posts cluster by modality,
not by vibe.

### Two fixes, both measured

1. **Re-weight the fusion** ([eval/sweep_fusion.py](eval/sweep_fusion.py)):
   sweeping the text weight found the optimum at 0.1, not 0.5 → **+29%** for a
   one-line change.
2. **Fuse in language instead of vector space**
   ([pipeline/generate_vibe_captions.py](pipeline/generate_vibe_captions.py)):
   a VLM (Gemini) reads each post's images + title + descriptors and writes one
   unified vibe paragraph, which a proper sentence encoder (bge-base) embeds.
   Captions alone rescued the broken text-only posts (0.2× → 2.3×) but lost
   visual nuance on image posts (5.5× vs ~7×) — so the final architecture
   keeps both:

```
post_vector = normalize([ 0.5 · bge(vibe_caption) ; 0.5 · siglip(images) ])
```

Cosine similarity on the concatenation is the weighted sum of the per-space
similarities, so the two spaces never need aligning — and text queries (a
future chat interface) search the caption block natively.

| Architecture (same population) | Rare-overlap lift |
|---|---|
| SigLIP 50/50 average (original) | 5.0× |
| SigLIP, swept weight (w_text = 0.1) | 5.9× |
| VLM captions alone (bge) | 4.8× |
| **Hybrid concat (final)** | **7.5×** |

All numbers: [eval/eval_results/](eval/eval_results/).

### Does it feel right? (movie-level spot checks)

Post vectors aggregate into **movie-level vibe vectors**
([pipeline/build_movie_vectors.py](pipeline/build_movie_vectors.py)) — each
movie is the weighted average of the posts that recommend it, with shotgun
50-movie threads down-weighted. Nearest neighbors in the hybrid space:

| Query | Nearest movies by vibe |
|---|---|
| *Lost in Translation* | Chungking Express · Her · Fallen Angels · All of Us Strangers · Eternal Sunshine |
| *The Thing* | Jacob's Ladder · The Void · In the Mouth of Madness · The Empty Man |
| *Fargo* | Winter's Bone · True Detective · The Place Beyond the Pines · Prisoners · Insomnia |
| *Paris, Texas* | No Country for Old Men · Nocturnal Animals · My Own Private Idaho |

None of these groupings follow genre tags — "neon urban loneliness" and
"snowbound crime bleakness" are not TMDB categories. That's the vibe space
working.

---

## Architecture

```
r/MoviesThatFeelLike (4,418 posts, images, 90k comments)
        │  scrape
        ▼
┌─ pipeline/ ───────────────────────────────────────────────┐
│ extract_and_verify_movies.py   Llama-3.1 pulls titles     │
│                                from comments → TMDB verify │
│ extract_vibe_tags_from_reddit  vibe descriptors per post   │
│ generate_vibe_captions.py      Gemini: images+text → one   │
│                                unified vibe paragraph      │
│ encode_modalities.py           SigLIP image vectors        │
│ embed_captions.py              bge-base caption vectors    │
│ build_hybrid_vectors.py        [0.5·caption ; 0.5·image]   │
│ build_movie_vectors.py         post → movie aggregation    │
└────────────────────────────────────────────────────────────┘
        │ 3,715 post vectors · 17,174 movie vectors
        ▼
┌─ eval/ ──────────────┐   ┌─ api/ ─────────────────────────┐
│ eval_embeddings.py   │   │ FastAPI /recommendations:      │
│ movie-overlap metric │   │ cosine retrieval + popularity  │
│ + eval_results/      │   │ debiasing (α slider)           │
└──────────────────────┘   └────────────┬───────────────────┘
                                        ▼
                           ┌─ frontend/ ────────────────────┐
                           │ Next.js: vibe-anchor grid,     │
                           │ penalty slider, movie grid     │
                           └────────────────────────────────┘
```

The retrieval layer includes **popularity debiasing**: a movie's score is
`local_count / global_count^α`, with α user-controlled — at α=0 you get the
crowd favorites, at α=1 the obscure Czech film that nails the vibe.

## Repo layout

| Path | What |
|---|---|
| `pipeline/` | Data pipeline, in run order (see Reproduce) |
| `eval/` | Eval harness + tracked results for every embedding version |
| `api/` | FastAPI retrieval backend |
| `frontend/` | Next.js demo (vibe anchors → recommendations) |
| `experiments/` | Clustering visualizations, scratch analysis |
| `data/` | All datasets/vectors/images — gitignored, regenerable |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # + a TMDB and Gemini key in pipeline/.env

# pipeline (each step caches; rerun-safe)
python pipeline/extract_and_verify_movies.py      # comments → verified movies (TMDB)
python pipeline/extract_vibe_tags_from_reddit.py  # descriptors (local Ollama)
python pipeline/generate_vibe_captions.py         # VLM vibe captions (Gemini, ~$5)
python pipeline/encode_modalities.py              # SigLIP image/text vectors
python pipeline/embed_captions.py                 # bge caption vectors
python pipeline/build_hybrid_vectors.py --eval    # final post vectors + eval
python pipeline/build_movie_vectors.py --posts-file data/posts_with_hybrid_vectors.json

# demo
uvicorn main:app --port 8000 --app-dir api        # backend
cd frontend && npm i && npm run dev               # http://localhost:3000
```

## Honest limitations

- **Corpus skew:** r/MoviesThatFeelLike leans atmospheric/moody — coverage is
  deep for "foggy seaside dread," thin for "fun date-night comedy." 4,792
  movies have ≥5 supporting posts; the 45% with a single post get noisy vectors.
- **Verification noise:** TMDB title matching has known misses (a 1978 thriller
  matched to a 2019 docuseries); the verify step needs a disambiguation pass.
- **Text-only posts** (8% of corpus) sit in a caption-only subspace and still
  underperform in the hybrid space — candidate fix is image-block imputation.
- The eval signal (movie co-recommendation) is a proxy for vibe similarity,
  not a human judgment — it's calibrated against a random baseline, but a
  small human-rated golden set is the natural next validation step.

## Roadmap

Next up (tracked in [STATUS.md](STATUS.md)): a contrastive projection head
trained on movie-overlap positives (the corpus supervises its own metric
learning), a text-query endpoint (bge-embed the user's words, search the
caption block), and the product's signature interaction — adaptive
image-probe rounds that triangulate your head-space through forced choices
between contrasting moods.
