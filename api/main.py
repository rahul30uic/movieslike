"""
Movieslike retrieval API — movie-level, hybrid vibe space.

Serves recommendations from movie vectors in the hybrid embedding space
(  [0.5 * bge(vibe_caption) ; 0.5 * siglip(images)]  , 1536-dim — see
eval/eval_results/hybrid_comparison.json for why).

Endpoints:
  POST /recommendations       — target_vector (1536-dim anchor) → movies
  POST /recommendations/text  — free-text vibe query → movies (bge-embedded,
                                searches the caption block of the hybrid space)

Popularity debiasing: score = cosine - alpha * 0.02 * log2(1 + n_posts),
where n_posts is how many Reddit posts recommended the movie. alpha=0 gives
crowd favorites; alpha=1 favors the long tail.

Run:  uvicorn main:app --port 8000 --app-dir api
"""

import csv
import json
import logging
import os
from typing import List

import io

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
MOVIE_VECTORS_FILE = os.path.join(DATA_DIR, "movie_vectors_hybrid.json")
POST_VECTORS_FILE = os.path.join(DATA_DIR, "posts_with_hybrid_vectors.json")
DATASET_CSV = os.path.join(DATA_DIR, "final_dataset.csv")

# --- Probe (head-space triangulation) parameters ---
PROBE_ROUNDS = 5
PROBE_SHARPNESS = 25.0   # Bradley-Terry k: how decisive one pick is
PROBE_CANDIDATES = 200   # candidate pairs sampled per round

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
IMAGE_MODEL = "google/siglip-base-patch16-224"  # must match pipeline/encode_modalities.py
# bge's recommended prefix for retrieval queries (captions are the passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_MIN_SUPPORT = 3  # movies below this post-support have noisy vectors
DEFAULT_MIN_VOTES = 0    # TMDB vote_count floor: recognizability, not vibe

app = FastAPI(
    title="Movieslike Retrieval API",
    description="Vibe-based movie recommendations from the hybrid embedding space.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VectorRequest(BaseModel):
    target_vector: List[float] = Field(..., description="1536-dim hybrid-space vector.")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Popularity penalty factor.")
    num_recommendations: int = Field(12, gt=0, le=50)
    min_support: int = Field(DEFAULT_MIN_SUPPORT, ge=1)
    min_votes: int = Field(DEFAULT_MIN_VOTES, ge=0, description="TMDB vote_count floor (recognizability).")
    top_k: int | None = Field(None, description="Unused; accepted for backward compatibility.")


class TextRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Free-text vibe description.")
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    num_recommendations: int = Field(12, gt=0, le=50)
    min_support: int = Field(DEFAULT_MIN_SUPPORT, ge=1)
    min_votes: int = Field(DEFAULT_MIN_VOTES, ge=0)


class Movie(BaseModel):
    tmdb_id: int
    title: str
    poster_path: str | None
    n_posts: int
    score: float


class RecommendationResponse(BaseModel):
    recommendations: List[Movie]


# Populated at startup
movies: List[dict] = []
movie_matrix: np.ndarray = np.array([])
supports: np.ndarray = np.array([])
penalties: np.ndarray = np.array([])
vote_counts: np.ndarray = np.array([])
embedder = None
image_encoder = None  # SigLIP, for query-by-image

# Probe pool: posts with a real image + caption, used as both the probe
# images shown to the user and the particles of the head-space posterior.
probe_posts: List[dict] = []       # {post_id, image_url, caption}
probe_matrix: np.ndarray = np.array([])
probe_index: dict = {}             # post_id -> row


@app.on_event("startup")
def load_everything():
    global movies, movie_matrix, supports, penalties, vote_counts, embedder, image_encoder

    npz_path = os.path.join(DATA_DIR, "movie_vectors_deploy.npz")
    meta_path = os.path.join(DATA_DIR, "movies_meta.json")
    if os.path.exists(npz_path) and os.path.exists(meta_path):
        # Compact deploy artifacts (pipeline/export_deploy_artifacts.py)
        logging.info(f"Loading movie vectors from {npz_path}...")
        movie_matrix = np.load(npz_path)["vectors"].astype(np.float32)
        with open(meta_path, encoding="utf-8") as f:
            movies.extend(json.loads(line) for line in f)
    elif os.path.exists(MOVIE_VECTORS_FILE):
        logging.info(f"Loading movie vectors from {MOVIE_VECTORS_FILE}...")
        vecs = []
        with open(MOVIE_VECTORS_FILE, encoding="utf-8") as f:
            for line in f:
                m = json.loads(line)
                vecs.append(m.pop("vector"))
                movies.append(m)
        movie_matrix = np.asarray(vecs, dtype=np.float32)
    else:
        raise FileNotFoundError(
            f"Neither {npz_path} nor {MOVIE_VECTORS_FILE} found — run "
            "pipeline/build_movie_vectors.py then pipeline/export_deploy_artifacts.py"
        )
    norms = np.linalg.norm(movie_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    movie_matrix /= norms
    supports = np.array([m["n_posts"] for m in movies])
    # Standardized corpus-popularity signal. Cosine scores are standardized
    # per query in rank(); with both z-scored, alpha is an honest dial:
    # 0 = pure vibe match, 1 = vibe and long-tail-ness weighted equally.
    log_pop = np.log2(1 + supports)
    penalties = (log_pop - log_pop.mean()) / (log_pop.std() + 1e-9)
    vote_counts = np.array([m.get("vote_count") or 0 for m in movies])
    logging.info(f"Loaded {len(movies)} movies ({movie_matrix.shape[1]}-dim).")

    # --- Probe pool (optional: requires local corpus images; the public
    # deployment ships without them and the probe endpoints return 503) ---
    global probe_matrix
    if not (os.path.exists(DATASET_CSV) and os.path.exists(POST_VECTORS_FILE)):
        logging.warning("Probe data not present — /probe endpoints disabled.")
        _finish_startup()
        return
    image_paths = {}
    with open(DATASET_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = (row.get("image_local_path") or "").split("|")[0].strip()
            if p and os.path.exists(os.path.join(DATA_DIR, p)):
                image_paths[row["post_id"]] = p
    probe_vecs = []
    with open(POST_VECTORS_FILE, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            rel = image_paths.get(p["post_id"])
            if rel and p.get("image_exists") and p.get("caption"):
                probe_index[p["post_id"]] = len(probe_posts)
                probe_posts.append({
                    "post_id": p["post_id"],
                    "image_url": "/" + rel,
                    "caption": p["caption"],
                })
                probe_vecs.append(p["combined_vector"])
    probe_matrix = np.asarray(probe_vecs, dtype=np.float32)
    logging.info(f"Probe pool: {len(probe_posts)} image posts.")
    _finish_startup()


def _finish_startup():
    global embedder, image_encoder
    logging.info(f"Loading {EMBED_MODEL} for text queries...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

    logging.info(f"Loading {IMAGE_MODEL} for image queries...")
    image_encoder = SentenceTransformer(IMAGE_MODEL, device="cpu")
    logging.info("Startup complete.")


def rank(target: np.ndarray, alpha: float, n: int, min_support: int,
         min_votes: int = DEFAULT_MIN_VOTES) -> List[Movie]:
    norm = np.linalg.norm(target)
    if norm == 0:
        raise HTTPException(status_code=422, detail="Target vector has zero norm.")
    sims = movie_matrix @ (target / norm)
    # Standardize: raw cosines cluster in a ~0.02 band, so an unstandardized
    # penalty would drown the vibe signal (the "everything is obscure" bug).
    z_sims = (sims - sims.mean()) / (sims.std() + 1e-9)
    scores = z_sims - alpha * penalties
    scores[supports < min_support] = -np.inf
    if min_votes > 0:
        scores[vote_counts < min_votes] = -np.inf

    top = np.argpartition(scores, -n)[-n:]
    top = top[np.argsort(-scores[top])]
    return [
        Movie(
            tmdb_id=movies[i]["tmdb_id"],
            title=movies[i]["title"],
            poster_path=movies[i].get("poster_path"),
            n_posts=int(supports[i]),
            score=round(float(scores[i]), 4),
        )
        for i in top
        if np.isfinite(scores[i])
    ]


@app.post("/recommendations", response_model=RecommendationResponse)
def recommend_by_vector(req: VectorRequest):
    if movie_matrix.size == 0:
        raise HTTPException(status_code=503, detail="Server not ready.")
    target = np.asarray(req.target_vector, dtype=np.float32)
    if target.shape[0] != movie_matrix.shape[1]:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {movie_matrix.shape[1]}-dim vector, got {target.shape[0]}. "
            "(Old 768-dim SigLIP anchors are incompatible with the hybrid space — "
            "regenerate frontend anchors.)",
        )
    return {"recommendations": rank(target, req.alpha, req.num_recommendations, req.min_support, req.min_votes)}


@app.post("/recommendations/text", response_model=RecommendationResponse)
def recommend_by_text(req: TextRequest):
    if embedder is None:
        raise HTTPException(status_code=503, detail="Server not ready.")
    q = embedder.encode(QUERY_PREFIX + req.query, normalize_embeddings=True)
    # Caption block only; zero image block. Uniform scaling of scores — ranking unaffected.
    target = np.concatenate([q, np.zeros_like(q)]).astype(np.float32)
    return {"recommendations": rank(target, req.alpha, req.num_recommendations, req.min_support, req.min_votes)}


@app.post("/recommendations/image", response_model=RecommendationResponse)
async def recommend_by_image(
    file: UploadFile = File(..., description="A mood image — screenshot, photo, painting."),
    alpha: float = Form(0.3),
    num_recommendations: int = Form(12),
    min_support: int = Form(DEFAULT_MIN_SUPPORT),
    min_votes: int = Form(DEFAULT_MIN_VOTES),
):
    """Query by image: SigLIP-embed the upload, search the image block."""
    if image_encoder is None:
        raise HTTPException(status_code=503, detail="Server not ready.")
    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 15MB).")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read that file as an image.")
    img.thumbnail((512, 512))  # SigLIP sees 224px; keep decode cheap

    v = image_encoder.encode(img, convert_to_numpy=True)
    v = v / (np.linalg.norm(v) + 1e-9)
    # Image block only; zero caption block — mirror of the text endpoint.
    target = np.concatenate([np.zeros_like(v), v]).astype(np.float32)
    n = max(1, min(num_recommendations, 50))
    return {"recommendations": rank(target, min(max(alpha, 0.0), 1.0), n, min_support, min_votes)}


# =============================================================================
# Head-space probes: pick-one-of-two mood images, Bayesian narrowing
# =============================================================================

class ProbeChoice(BaseModel):
    chosen: str = Field(..., description="post_id of the picked image")
    rejected: str = Field(..., description="post_id of the other image")


class ProbeNextRequest(BaseModel):
    history: List[ProbeChoice] = []


class ProbeImage(BaseModel):
    post_id: str
    image_url: str
    caption: str


class ProbeNextResponse(BaseModel):
    round: int
    total_rounds: int
    pair: List[ProbeImage]


class ProbeRecommendRequest(BaseModel):
    history: List[ProbeChoice] = Field(..., min_length=1)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    num_recommendations: int = Field(5, gt=0, le=12)
    min_support: int = Field(DEFAULT_MIN_SUPPORT, ge=1)
    min_votes: int = Field(DEFAULT_MIN_VOTES, ge=0)


def probe_posterior(history: List[ProbeChoice]) -> np.ndarray:
    """
    Particle posterior over the probe pool. Each pick reweights every particle
    by the Bradley-Terry likelihood that a user 'located at' that particle
    would have made the same pick.
    """
    w = np.ones(len(probe_posts), dtype=np.float64)
    for h in history:
        ci, ri = probe_index.get(h.chosen), probe_index.get(h.rejected)
        if ci is None or ri is None:
            raise HTTPException(status_code=422, detail=f"Unknown post_id in history: {h}")
        margin = probe_matrix @ probe_matrix[ci] - probe_matrix @ probe_matrix[ri]
        w *= 1.0 / (1.0 + np.exp(-PROBE_SHARPNESS * margin))
    total = w.sum()
    if total <= 0:
        return np.ones(len(probe_posts)) / len(probe_posts)
    return w / total


def head_space_vector(history: List[ProbeChoice]) -> np.ndarray:
    w = probe_posterior(history)
    v = (w[:, None] * probe_matrix).sum(axis=0)
    return v.astype(np.float32)


@app.post("/probe/next", response_model=ProbeNextResponse)
def probe_next(req: ProbeNextRequest):
    if probe_matrix.size == 0:
        raise HTTPException(status_code=503, detail="Server not ready.")

    w = probe_posterior(req.history)
    shown = {pid for h in req.history for pid in (h.chosen, h.rejected)}
    available = np.array([i for i, p in enumerate(probe_posts) if p["post_id"] not in shown])
    rng = np.random.default_rng()  # fresh entropy: every session sees different probes

    # Anchor image A: sampled from the posterior (a plausible head-space).
    wa = w[available]
    wa = wa / wa.sum()
    a = int(rng.choice(available, p=wa))

    # Contrast image B: among posterior-weighted candidates, prefer pairs that
    # are far apart AND that the posterior can't predict (split ~50/50) —
    # a cheap proxy for expected information gain.
    cand = rng.choice(available, size=min(PROBE_CANDIDATES, len(available)), replace=False, p=wa)
    cand = cand[cand != a]
    sim_ab = probe_matrix[cand] @ probe_matrix[a]
    # Predicted probability the user picks A over each candidate B
    margins = (probe_matrix @ probe_matrix[a])[None, :] - (probe_matrix[cand] @ probe_matrix.T)
    p_a = (w[None, :] / (1.0 + np.exp(-PROBE_SHARPNESS * margins))).sum(axis=1)
    balance_penalty = np.abs(p_a - 0.5)
    b = int(cand[np.argmin(balance_penalty + 0.5 * sim_ab)])

    pair = [probe_posts[a], probe_posts[b]]
    return {
        "round": len(req.history) + 1,
        "total_rounds": PROBE_ROUNDS,
        "pair": pair,
    }


@app.post("/probe/recommend", response_model=RecommendationResponse)
def probe_recommend(req: ProbeRecommendRequest):
    if probe_matrix.size == 0:
        raise HTTPException(status_code=503, detail="Server not ready.")
    target = head_space_vector(req.history)
    return {"recommendations": rank(target, req.alpha, req.num_recommendations, req.min_support, req.min_votes)}


@app.get("/")
def root():
    return {
        "message": "Movieslike retrieval API (hybrid vibe space).",
        "movies": len(movies),
        "endpoints": ["/recommendations", "/recommendations/text", "/recommendations/image", "/probe/next", "/probe/recommend"],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
