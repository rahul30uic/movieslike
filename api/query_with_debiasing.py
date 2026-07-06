import os
import json
import logging
from collections import Counter
from typing import List, Dict, Optional

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone

# =============================================================================
# Configuration
# =============================================================================
# --- Dependencies ---
# You'll need to install the following packages:
# pip install pinecone-client numpy python-dotenv

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- File Paths & Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
VECTORS_INPUT_JSONL = os.path.join(DATA_DIR, "posts_with_vectors.json")

# --- Pinecone Configuration ---
INDEX_NAME = "movie-vibes"
VECTOR_DIMENSION = 768

# =============================================================================
# Core Functions
# =============================================================================

def build_global_frequency_map(filepath: str) -> Counter:
    """
    Reads the entire dataset and builds a global frequency map of all movie IDs.

    Args:
        filepath: Path to the 'posts_with_vectors.json' JSONL file.

    Returns:
        A Counter object mapping each movie_id (int) to its total count.
    """
    logging.info(f"Building global movie frequency map from {filepath}...")
    global_freq_counter = Counter()
    
    if not os.path.exists(filepath):
        logging.error(f"Input file not found: {filepath}")
        return global_freq_counter

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                post = json.loads(line)
                # 'tmdb_ids' in the source file are integers
                if 'tmdb_ids' in post and isinstance(post['tmdb_ids'], list):
                    global_freq_counter.update(post['tmdb_ids'])
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed line in {filepath}")
                continue
                
    logging.info(f"Global frequency map built. Found {len(global_freq_counter)} unique movies.")
    return global_freq_counter


def get_debiased_movie_recommendations(
    pinecone_index: Pinecone.Index,
    global_freq_map: Counter,
    target_vector: List[float],
    top_k: int = 30,
    alpha: float = 0.5,
    num_recommendations: int = 5
) -> List[str]:
    """
    Queries Pinecone and re-ranks results using a popularity discount factor.

    Args:
        pinecone_index: The initialized Pinecone index object.
        global_freq_map: A Counter of global movie frequencies.
        target_vector: The 768-dimensional vector to search for.
        top_k: The initial number of candidates to fetch from Pinecone.
        alpha: The penalty factor. 0.0 = standard popularity, 1.0 = full penalty.
        num_recommendations: The final number of movie IDs to return.

    Returns:
        A list of the top re-ranked movie IDs.
    """
    logging.info(f"Querying Pinecone for top {top_k} candidates with alpha={alpha}...")

    # --- Step 1: Query Pinecone for a candidate pool ---
    query_response = pinecone_index.query(
        vector=target_vector,
        top_k=top_k,
        include_metadata=True
    )

    # --- Step 2: Calculate local frequency in the neighborhood ---
    all_local_movies = []
    for match in query_response.get('matches', []):
        # 'recommended_movies' in Pinecone metadata are strings
        recommended_ids = match.get('metadata', {}).get('recommended_movies', [])
        if recommended_ids:
            all_local_movies.extend(recommended_ids)
    
    if not all_local_movies:
        logging.warning("No recommended movies found in the query results' metadata.")
        return []

    local_freq = Counter(all_local_movies)
    logging.info(f"Found {len(local_freq)} unique movies in the local neighborhood.")

    # --- Step 3: Apply the popularity discount formula ---
    scored_movies = []
    for movie_id_str, local_count in local_freq.items():
        try:
            movie_id_int = int(movie_id_str)
            # Default to 1 to avoid division by zero for movies not in the global map
            global_count = global_freq_map.get(movie_id_int, 1)
            
            # Score = Local_Frequency / (Global_Frequency ** alpha)
            score = local_count / (global_count ** alpha)
            scored_movies.append((movie_id_str, score))
        except (ValueError, TypeError):
            logging.warning(f"Could not process movie ID '{movie_id_str}'. Skipping.")
            continue

    # --- Step 4: Sort by adjusted score and return top recommendations ---
    scored_movies.sort(key=lambda x: x[1], reverse=True)
    
    top_movie_ids = [movie_id for movie_id, score in scored_movies[:num_recommendations]]
    
    return top_movie_ids

# =============================================================================
# Testing Suite
# =============================================================================

if __name__ == "__main__":
    # --- 1. Setup ---
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found. Please set it in your .env file.")

    pc = Pinecone(api_key=api_key)
    
    if INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(f"Index '{INDEX_NAME}' does not exist. Please run 'upsert_to_pinecone.py' first.")
        
    index = pc.Index(INDEX_NAME)
    
    # --- 2. Build Global Frequency Map ---
    global_frequencies = build_global_frequency_map(VECTORS_INPUT_JSONL)

    # --- 3. Generate a Mock Vector and Run Queries ---
    mock_vector = np.random.rand(VECTOR_DIMENSION).tolist()

    # Run with standard popularity (alpha=0.0)
    standard_recs = get_debiased_movie_recommendations(index, global_frequencies, mock_vector, alpha=0.0)

    # Run with de-biasing penalty (alpha=0.6)
    debiased_recs = get_debiased_movie_recommendations(index, global_frequencies, mock_vector, alpha=0.6)

    # --- 4. Print Results for Comparison ---
    print("\n" + "="*60)
    print("           Recommendation Comparison")
    print("="*60)
    print(f"Standard Popularity (alpha=0.0): {standard_recs}")
    print(f"De-Biased Results (alpha=0.6):   {debiased_recs}")
    print("="*60)
    print("\nNotice how the de-biased list likely contains different, less globally-dominant movies.")
