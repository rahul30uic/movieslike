import os
import json
import logging
from collections import Counter
from typing import List, Dict, Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# Configuration
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
VECTORS_INPUT_JSONL = os.path.join(DATA_DIR, "posts_with_vectors.json")
# This file contains the full movie metadata including poster paths
MOVIES_INPUT_JSON = os.path.join(DATA_DIR, "phase1_extracted_movies_fresh_run.json")

# =============================================================================
# FastAPI App Initialization
# =============================================================================
app = FastAPI(
    title="MovieVibes AI Backend",
    description="API for serving movie recommendations based on multimodal embeddings.",
    version="1.0.0",
)

# --- CORS Middleware ---
# Allows the frontend (running on localhost:3000) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Adjust if your frontend runs on a different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Data Models
# =============================================================================
class RecommendationRequest(BaseModel):
    target_vector: List[float] = Field(..., description="The 768-dimensional vector of the selected vibe anchor.")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="The popularity penalty factor. 0.0 = standard popularity, 1.0 = full penalty.")
    top_k: int = Field(30, gt=0, description="The initial number of candidates to fetch for re-ranking.")
    num_recommendations: int = Field(12, gt=0, description="The final number of movie recommendations to return.")

class Movie(BaseModel):
    tmdb_id: int
    title: str
    poster_path: str | None

class RecommendationResponse(BaseModel):
    recommendations: List[Movie]

# =============================================================================
# In-Memory Data Store
# =============================================================================
# These will be populated at startup
posts_data: List[Dict[str, Any]] = []
all_vectors: np.ndarray = np.array([])
global_freq_map: Counter = Counter()
movie_details_map: Dict[int, Dict[str, Any]] = {}

# =============================================================================
# Loading and Pre-computation Logic (at startup)
# =============================================================================
def load_data_and_build_maps():
    """Loads all necessary data from JSON files into memory."""
    global posts_data, all_vectors, global_freq_map, movie_details_map
    logging.info("Starting data loading process...")

    # --- 1. Load posts with vectors ---
    if not os.path.exists(VECTORS_INPUT_JSONL):
        raise FileNotFoundError(f"Vector data file not found: {VECTORS_INPUT_JSONL}")
    
    logging.info(f"Loading posts and vectors from {VECTORS_INPUT_JSONL}...")
    temp_posts = []
    temp_vectors = []
    with open(VECTORS_INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                post = json.loads(line)
                # Ensure the post has the required fields
                if 'combined_vector' in post and 'tmdb_ids' in post:
                    temp_posts.append(post)
                    temp_vectors.append(post['combined_vector'])
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed line in {VECTORS_INPUT_JSONL}")
    
    posts_data = temp_posts
    all_vectors = np.array(temp_vectors, dtype=np.float32)
    logging.info(f"Loaded {len(posts_data)} posts and their vectors.")

    # --- 2. Build global movie frequency map ---
    logging.info("Building global movie frequency map...")
    all_tmdb_ids = []
    for post in posts_data:
        # tmdb_ids are integers in the source file
        if isinstance(post.get('tmdb_ids'), list):
            all_tmdb_ids.extend(post['tmdb_ids'])
    global_freq_map = Counter(all_tmdb_ids)
    logging.info(f"Global frequency map built. Found {len(global_freq_map)} unique movies.")

    # --- 3. Build movie details map for quick lookups ---
    if not os.path.exists(MOVIES_INPUT_JSON):
        raise FileNotFoundError(f"Movie metadata file not found: {MOVIES_INPUT_JSON}")

    logging.info(f"Building movie details map from {MOVIES_INPUT_JSON}...")
    with open(MOVIES_INPUT_JSON, 'r', encoding='utf-8') as f:
        phase1_data = json.load(f)
    
    for post in phase1_data:
        recommendations = post.get("verified_recommendations", [])
        if isinstance(recommendations, list):
            for rec in recommendations:
                if isinstance(rec, dict) and 'tmdb_id' in rec:
                    tmdb_id = rec['tmdb_id']
                    if tmdb_id not in movie_details_map:
                        movie_details_map[tmdb_id] = {
                            "title": rec.get("official_title", "Unknown Title"),
                            "poster_path": rec.get("poster_path")
                        }
    logging.info(f"Movie details map built. Found details for {len(movie_details_map)} unique movies.")
    logging.info("Startup data loading complete.")


@app.on_event("startup")
async def startup_event():
    """FastAPI startup event handler."""
    load_data_and_build_maps()

# =============================================================================
# API Endpoints
# =============================================================================
@app.post("/recommendations", response_model=RecommendationResponse)
async def get_debiased_recommendations(request: RecommendationRequest):
    """
    Accepts a target vector and returns a list of debiased movie recommendations.
    """
    if not posts_data or all_vectors.size == 0:
        raise HTTPException(status_code=503, detail="Server is not ready, data not loaded.")

    target_vector = np.array(request.target_vector).reshape(1, -1)

    # --- Step 1: Find the candidate pool using cosine similarity ---
    logging.info(f"Finding top {request.top_k} candidates...")
    similarities = cosine_similarity(target_vector, all_vectors)[0]
    
    # Get indices of the top_k most similar posts
    top_k_indices = np.argpartition(similarities, -request.top_k)[-request.top_k:]
    
    # --- Step 2: Calculate local frequency in the neighborhood ---
    local_movie_ids = []
    for idx in top_k_indices:
        post = posts_data[idx]
        if isinstance(post.get('tmdb_ids'), list):
            local_movie_ids.extend(post['tmdb_ids'])
    
    if not local_movie_ids:
        logging.warning("No recommended movies found in the candidate pool.")
        return {"recommendations": []}

    local_freq = Counter(local_movie_ids)
    logging.info(f"Found {len(local_freq)} unique movies in the local neighborhood.")

    # --- Step 3: Apply the popularity discount formula ---
    scored_movies = []
    for movie_id, local_count in local_freq.items():
        global_count = global_freq_map.get(movie_id, 1)
        score = local_count / (global_count ** request.alpha)
        scored_movies.append((movie_id, score))

    # --- Step 4: Sort by adjusted score and get top recommendations ---
    scored_movies.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [movie_id for movie_id, score in scored_movies[:request.num_recommendations]]

    # --- Step 5: Format the response with movie details ---
    recommendations = []
    for movie_id in top_movie_ids:
        details = movie_details_map.get(movie_id)
        if details:
            recommendations.append(
                Movie(tmdb_id=movie_id, title=details['title'], poster_path=details['poster_path'])
            )
        else:
            logging.warning(f"Could not find details for movie ID {movie_id}. Skipping.")

    return {"recommendations": recommendations}

@app.get("/")
async def root():
    return {"message": "MovieVibes AI Backend is running. Use the /recommendations endpoint."}

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)