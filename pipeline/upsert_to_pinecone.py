import os
import json
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Iterator, Dict, Any, List
import logging

# =============================================================================
# Configuration
# =============================================================================
# --- Dependencies ---
# You'll need to install the following packages:
# pip install pinecone-client pandas python-dotenv tqdm

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- File Paths & Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
VECTORS_INPUT_JSONL = os.path.join(DATA_DIR, "posts_with_vectors.json")

# --- Pinecone Configuration ---
INDEX_NAME = "movie-vibes"
DIMENSION = 768
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"
BATCH_SIZE = 100

# =============================================================================
# Helper Functions
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the posts with their vectors from a JSONL file.

    Args:
        filepath: The path to the input JSONL file.

    Returns:
        A pandas DataFrame with the loaded data.
    """
    logging.info(f"Loading vector data from {filepath}...")
    if not os.path.exists(filepath):
        logging.error(f"Input file not found: {filepath}")
        logging.error("Please run 'generate_embeddings_and_clusters.py' first to create this file.")
        return pd.DataFrame()
    try:
        df = pd.read_json(filepath, lines=True)
        logging.info(f"Successfully loaded {len(df)} posts with vectors.")
        return df
    except Exception as e:
        logging.error(f"Failed to load or parse {filepath}: {e}")
        return pd.DataFrame()

def format_for_pinecone(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Formats the DataFrame records into a list of dictionaries compatible with Pinecone.

    Args:
        df: The input DataFrame containing post data and vectors.

    Returns:
        A list of dictionaries ready for upserting.
    """
    pinecone_vectors = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Formatting Payloads"):
        # Ensure tmdb_ids are converted to a list of strings for metadata compatibility
        tmdb_ids_as_strings = []
        if isinstance(row.get('tmdb_ids'), list):
            tmdb_ids_as_strings = [str(tmdb_id) for tmdb_id in row['tmdb_ids']]

        # Construct the metadata dictionary
        metadata = {
            "cluster_id": int(row.get('cluster_id', -1)),
            "image_path": str(row.get('image_local_path', '')),
            "vibe_tags": row.get('descriptors', []),
            "recommended_movies": tmdb_ids_as_strings
        }

        # Create the final vector payload
        pinecone_vectors.append({
            "id": str(row['post_id']),
            "values": row['combined_vector'],
            "metadata": metadata
        })
    return pinecone_vectors

def yield_batches(vectors: List[Dict[str, Any]], batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    """
    Yields batches of vectors to be upserted.

    Args:
        vectors: A list of all vectors to be upserted.
        batch_size: The size of each batch.

    Yields:
        A chunk of vectors of the specified batch size.
    """
    for i in range(0, len(vectors), batch_size):
        yield vectors[i:i + batch_size]

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to orchestrate the Pinecone upsert pipeline."""
    # --- 1. Initialization ---
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables. Please create a .env file and add it.")

    pc = Pinecone(api_key=api_key)

    # --- 2. Create Index if it doesn't exist ---
    if INDEX_NAME not in pc.list_indexes().names():
        logging.info(f"Index '{INDEX_NAME}' not found. Creating a new serverless index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION)
        )
        logging.info("Index created successfully.")
    else:
        logging.info(f"Index '{INDEX_NAME}' already exists. Connecting to it.")

    index = pc.Index(INDEX_NAME)
    logging.info(index.describe_index_stats())

    # --- 3. Load and Format Data ---
    df = load_data(VECTORS_INPUT_JSONL)
    if df.empty:
        return

    pinecone_payload = format_for_pinecone(df)

    # --- 4. Batch Upsert Data ---
    logging.info(f"Starting to upsert {len(pinecone_payload)} vectors in batches of {BATCH_SIZE}...")
    for batch in tqdm(yield_batches(pinecone_payload, BATCH_SIZE), total=(len(pinecone_payload) + BATCH_SIZE - 1) // BATCH_SIZE):
        try:
            index.upsert(vectors=batch)
        except Exception as e:
            logging.error(f"An error occurred during batch upsert: {e}")
            logging.error("Skipping this batch and continuing...")

    logging.info("Upsert process finished.")
    logging.info(index.describe_index_stats())

if __name__ == "__main__":
    main()