import pandas as pd
import numpy as np
import os
import logging
import ast
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import hdbscan
from tqdm import tqdm
import json
from typing import List, Dict, Any, Tuple

# =============================================================================
# Configuration
# =============================================================================
# --- Dependencies ---
# You'll need to install the following packages:
# pip install pandas numpy Pillow sentence-transformers scikit-learn hdbscan tqdm

# --- File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
INPUT_CSV = os.path.join(DATA_DIR, "final_dataset.csv")
ANCHORS_OUTPUT_JSON = os.path.join(DATA_DIR, "landing_grid_anchors.json")
VECTORS_OUTPUT_JSON = os.path.join(DATA_DIR, "posts_with_vectors.json")

# --- Model & Processing Parameters ---
BATCH_SIZE = 128
MODEL_NAME = 'google/siglip-base-patch16-224'
DEVICE = 'mps'  # Use Apple Silicon GPU

# --- Clustering Parameters ---
MIN_CLUSTER_SIZE = 30
MIN_SAMPLES = 10

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper Class for JSON Serialization ---
class NpEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy types. """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file and safely parses stringified columns.
    
    Args:
        filepath: The path to the input CSV file.

    Returns:
        A pandas DataFrame with cleaned and parsed data.
    """
    logging.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Safely parse stringified lists.
    for col in ['descriptors', 'verified_movies', 'tmdb_ids']:
        if col in df.columns:
            # Ensure NaNs are handled, otherwise ast.literal_eval will fail.
            def safe_literal_eval(val):
                if pd.isna(val):
                    return []
                try:
                    return ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    logging.warning(f"Malformed string in column '{col}': '{val}'. Returning empty list.")
                    return []
            df[col] = df[col].apply(safe_literal_eval)
            
    logging.info(f"Loaded {len(df)} posts.")
    return df


def generate_embeddings(df: pd.DataFrame, model: SentenceTransformer, batch_size: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generates multimodal embeddings for each post by combining image and text vectors.
    This version handles multiple images per post by averaging their embeddings.

    Args:
        df: The input DataFrame containing post data.
        model: The pre-trained SentenceTransformer (CLIP) model.
        batch_size: The number of items to process in each batch.

    Returns:
        A tuple containing:
        - A DataFrame of the posts that were successfully processed.
        - A NumPy array of the combined, normalized vectors.
    """
    logging.info("Generating multimodal embeddings...")

    # --- Robust Path Handling ---
    # Assume the image folders are located within the same directory as the script.
    project_root = DATA_DIR

    def get_all_valid_paths(path_str: str, root_dir: str) -> List[str]:
        """Parses a pipe-separated string of paths and returns a list of valid, absolute paths."""
        if not isinstance(path_str, str):
            return []
        
        relative_paths = [p.strip() for p in path_str.split('|')]
        absolute_paths = [os.path.join(root_dir, p) for p in relative_paths]
        
        # Filter for paths that actually exist on the filesystem
        valid_paths = [p for p in absolute_paths if os.path.exists(p)]
        
        if len(valid_paths) < len(absolute_paths):
            missing = set(absolute_paths) - set(valid_paths)
            logging.debug(f"Skipped {len(missing)} non-existent image paths from entry: '{path_str}'. Missing: {missing}")
            
        return valid_paths

    df['abs_image_paths'] = df['image_local_path'].apply(lambda p: get_all_valid_paths(p, project_root))

    logging.info("Verifying image paths...")
    df['image_exists'] = df['abs_image_paths'].apply(lambda paths: isinstance(paths, list) and len(paths) > 0)

    invalid_paths_df = df[~df['image_exists']]
    missing_files_count = len(invalid_paths_df)
    if missing_files_count > 0:
        # Clarified the warning message to indicate posts are processed as text-only.
        logging.warning(f"Found {missing_files_count} posts with missing or invalid image paths. They will be processed as text-only.")
        # Log the first few invalid paths to help with debugging
        for _, row in invalid_paths_df.head(5).iterrows():
            path_str = row['image_local_path']
            # Specifically check for NaN values, which are expected for text-only posts.
            if pd.isna(path_str):
                logging.warning(f"  - Example text-only post (ID: {row['post_id']}). No image path provided.")
            elif isinstance(path_str, str) and path_str.strip():
                first_rel_path = path_str.split('|')[0].strip()
                first_abs_path_check = os.path.join(project_root, first_rel_path)
                logging.warning(f"  - Example post with no valid images (ID: {row['post_id']}). First path checked: '{first_abs_path_check}' (Exists: {os.path.exists(first_abs_path_check)})")
            else:
                logging.warning(f"  - Example post with empty image path string (ID: {row['post_id']}): '{path_str}'")
        logging.info(f"Image path root used for resolution: {project_root}")
        logging.info("This is expected for text-only posts. For others, ensure image folders are in the 'movie extraction' directory.")

    # We will now process all posts, not just those with images.
    # The logic inside the loop will handle text-only, image-only, and multimodal posts.
    logging.info(f"Processing all {len(df)} posts for embedding generation.")

    all_combined_vectors = []
    processed_df = df.copy() # Use a copy to avoid modifying the original df in place outside this function scope.

    for i in tqdm(range(0, len(processed_df), batch_size), desc="Encoding Batches"):
        batch_df = processed_df.iloc[i:i + batch_size]
        
        # --- Text Encoding ---
        batch_texts = [" ".join(tags) if isinstance(tags, list) and tags else "" for tags in batch_df['descriptors']]
        text_vectors = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)

        # --- Image Encoding ---
        all_batch_images_flat = []
        image_to_post_map = []  # Maps each image in the flat list to its post's index within the batch
        for post_batch_idx, img_paths in enumerate(batch_df['abs_image_paths']):
            for img_path in img_paths:
                try:
                    all_batch_images_flat.append(Image.open(img_path).convert("RGB"))
                    image_to_post_map.append(post_batch_idx)
                except Exception as e:
                    logging.warning(f"Could not open image {img_path}, it will be skipped. Error: {e}")

        img_vectors_flat = model.encode(all_batch_images_flat, convert_to_numpy=True, show_progress_bar=False) if all_batch_images_flat else np.array([])

        # --- Combine Vectors for each post in the batch ---
        for post_batch_idx in range(len(batch_df)):
            text_vec_norm = normalize(text_vectors[post_batch_idx].reshape(1, -1))[0]
            
            # Find all image vectors corresponding to the current post
            post_img_indices = [k for k, p_idx in enumerate(image_to_post_map) if p_idx == post_batch_idx]
            
            if post_img_indices:
                post_img_vectors = img_vectors_flat[post_img_indices]
                post_img_vectors_norm = normalize(post_img_vectors, axis=1)
                avg_img_vec_norm = np.mean(post_img_vectors_norm, axis=0)
                # Re-normalize the averaged vector
                avg_img_vec_norm = normalize(avg_img_vec_norm.reshape(1, -1))[0]
            else:
                avg_img_vec_norm = None # No images were loaded for this post

            # Combine based on what's available
            if avg_img_vec_norm is not None and batch_texts[post_batch_idx].strip():
                avg_vec = (0.5 * avg_img_vec_norm) + (0.5 * text_vec_norm)
                combined_vec = normalize(avg_vec.reshape(1, -1))[0]
            elif avg_img_vec_norm is not None:
                combined_vec = avg_img_vec_norm
            else: # Only text is available (or neither)
                combined_vec = text_vec_norm
                
            all_combined_vectors.append(combined_vec)

    if len(processed_df) != len(all_combined_vectors):
        raise RuntimeError("Mismatch between the number of processed posts and generated vectors. This indicates a bug.")
        
    return processed_df, np.array(all_combined_vectors)


def cluster_and_find_anchors(df: pd.DataFrame, vectors: np.ndarray) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Performs HDBSCAN clustering and identifies the anchor post for each cluster.

    Args:
        df: The DataFrame of processed posts.
        vectors: The NumPy array of embeddings corresponding to the DataFrame.

    Returns:
        A tuple containing:
        - The DataFrame with an added 'cluster_id' column.
        - A list of dictionaries, where each dictionary represents an anchor post.
    """
    logging.info(f"Running HDBSCAN with min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}, and method='leaf'...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_method='leaf',
        metric='euclidean'
    )
    cluster_labels = clusterer.fit_predict(vectors)
    df['cluster_id'] = cluster_labels
    
    logging.info(f"Found {len(np.unique(cluster_labels)) - 1} clusters. {np.sum(cluster_labels == -1)} posts are outliers.")

    anchor_posts = []
    # Iterate through each valid cluster (ignore -1 outliers)
    for cluster_id in tqdm(np.unique(cluster_labels[cluster_labels != -1]), desc="Finding Cluster Anchors"):
        cluster_indices = df[df['cluster_id'] == cluster_id].index
        cluster_vectors = vectors[cluster_indices]
        
        # Calculate the centroid (geometric mean) of the cluster
        centroid = np.mean(cluster_vectors, axis=0)
        
        # Find the post closest to the centroid
        distances = pairwise_distances(cluster_vectors, centroid.reshape(1, -1), metric='euclidean')
        closest_index_in_cluster = np.argmin(distances)
        
        # Get the original DataFrame index for the anchor post
        anchor_post_original_index = cluster_indices[closest_index_in_cluster]
        anchor_post_series = df.loc[anchor_post_original_index]
        
        anchor_posts.append(anchor_post_series.to_dict())
        
    return df, anchor_posts


def main():
    """Main function to orchestrate the embedding and clustering pipeline."""
    # --- 1. Load and Initialize ---
    df = load_and_prepare_data(INPUT_CSV)
    logging.info(f"Initializing SentenceTransformer model '{MODEL_NAME}' on device '{DEVICE}'...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # --- 2. Generate Embeddings ---
    processed_df, vectors = generate_embeddings(df, model, BATCH_SIZE)
    # Add the combined vector to the DataFrame before finding anchors
    processed_df['combined_vector'] = [v.tolist() for v in vectors]

    if vectors.shape[0] == 0:
        logging.error("No embeddings were generated. Please check your data and image paths. Exiting.")
        return

    # --- 3. Cluster and Find Anchors ---
    df_with_clusters, anchor_posts = cluster_and_find_anchors(processed_df, vectors)

    # --- 4. Prepare and Save Outputs ---
    logging.info(f"Saving {len(anchor_posts)} anchor posts to {ANCHORS_OUTPUT_JSON}...")
    with open(ANCHORS_OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(anchor_posts, f, indent=4, cls=NpEncoder)

    logging.info(f"Saving {len(df_with_clusters)} posts with vectors and clusters to {VECTORS_OUTPUT_JSON}...")
    # Use pandas to_json for robust serialization
    df_with_clusters.to_json(VECTORS_OUTPUT_JSON, orient='records', lines=True, default_handler=str)

    logging.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
