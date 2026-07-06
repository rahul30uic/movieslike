import pandas as pd
import json
import os
import logging

# Configure logging for clear status updates and error messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
# Input from the main movie verification pipeline (Phase 1)
MOVIES_INPUT_JSON = os.path.join(DATA_DIR, "phase1_extracted_movies_fresh_run.json") # Using the correct 'fresh_run' file as you pointed out.
# Input from the vibe tag extraction pipeline (Phase 2)
VIBES_INPUT_JSON = os.path.join(DATA_DIR, "phase2_vibe_tags_output.json")
# NEW: Input for the original Reddit data to get the correct image path
REDDIT_DATA_CSV = os.path.join(DATA_DIR, "Reddit_data.csv")
# Output file for the combined data
OUTPUT_CSV = os.path.join(DATA_DIR, "final_dataset.csv")

def create_final_dataframe():
    """
    Combines data from the movie verification, vibe tag extraction, and original Reddit data
    into a single Pandas DataFrame and saves it as a CSV.
    """
    # --- 1. Load Data ---
    logging.info(f"Loading movie data from {MOVIES_INPUT_JSON}...")
    if not os.path.exists(MOVIES_INPUT_JSON):
        logging.error(f"Movie data file not found: {MOVIES_INPUT_JSON}")
        logging.error("Please run the main pipeline (e.g., extract_and_verify_movies.py) first.")
        return None
    try:
        df_movies = pd.read_json(MOVIES_INPUT_JSON)
    except Exception as e:
        logging.error(f"Failed to load or parse {MOVIES_INPUT_JSON}: {e}")
        return None

    logging.info(f"Loading vibe tag data from {VIBES_INPUT_JSON}...")
    if not os.path.exists(VIBES_INPUT_JSON):
        logging.error(f"Vibe tag data file not found: {VIBES_INPUT_JSON}")
        return None
    try:
        df_vibes = pd.read_json(VIBES_INPUT_JSON)
    except Exception as e:
        logging.error(f"Failed to load or parse {VIBES_INPUT_JSON}: {e}")
        return None

    logging.info(f"Loading Reddit data from {REDDIT_DATA_CSV}...")
    if not os.path.exists(REDDIT_DATA_CSV):
        logging.error(f"Reddit data file not found: {REDDIT_DATA_CSV}")
        return None
    try:
        df_reddit = pd.read_csv(REDDIT_DATA_CSV)
        # Strip leading/trailing whitespace from column names to prevent KeyErrors
        df_reddit.columns = df_reddit.columns.str.strip()

        # Rename columns from Reddit_data.csv to match the expected names for merging.
        # 'id' -> 'post_id', 'image_local_paths' -> 'image_local_path'
        df_reddit = df_reddit.rename(columns={'id': 'post_id', 'image_local_paths': 'image_local_path'})

        # Validate that required columns exist in df_reddit
        required_reddit_cols = ['post_id', 'image_local_path']
        if not all(col in df_reddit.columns for col in required_reddit_cols):
            logging.error(f"'{REDDIT_DATA_CSV}' is missing one of the required columns: {required_reddit_cols} (even after renaming). Found columns: {df_reddit.columns.tolist()}")
            return None
    except Exception as e:
        logging.error(f"Failed to load or parse {REDDIT_DATA_CSV}: {e}")
        return None

    # --- 2. Merge DataFrames ---
    logging.info("Merging movie and vibe tag data on 'post_id'...")
    # Using a 'left' merge to keep all posts from the movie extraction, even if they don't have descriptors.
    df_final = pd.merge(df_movies, df_vibes, on='post_id', how='left')

    logging.info("Merging with Reddit data to get correct image_local_path...")
    # Using a 'left' merge here as well to ensure no data is lost from the primary movie dataframe.
    df_final = pd.merge(df_final, df_reddit[['post_id', 'image_local_path']], on='post_id', how='left')

    if df_final.empty:
        logging.warning("The final DataFrame is empty. Check if 'post_id' values match between the input files.")
        return None

    # --- 3. Create Final Columns ---
    logging.info("Processing and creating final columns...")

    # Rename the correct vibe tags column and drop the conflicting/incorrect ones.
    # This addresses both the empty 'descriptors' and the incorrect 'image_local_path' issues.
    df_final = df_final.rename(columns={'vibe_tags_y': 'descriptors'})
    # The 'vibe_tags_x' column from df_movies is incorrect/empty, and 'image_url' is the post_url, not the path.
    # We drop them as they are no longer needed.
    if 'vibe_tags_x' in df_final.columns and 'image_url' in df_final.columns:
        df_final = df_final.drop(columns=['vibe_tags_x', 'image_url'])

    def extract_from_recs(recs, key):
        if not isinstance(recs, list):
            return []
        seen_ids = set()
        unique_recs = []
        for rec in recs:
            if isinstance(rec, dict) and rec.get('tmdb_id') not in seen_ids:
                unique_recs.append(rec)
                seen_ids.add(rec.get('tmdb_id'))
        return [rec.get(key) for rec in unique_recs]

    df_final['verified_movies'] = df_final['verified_recommendations'].apply(lambda r: extract_from_recs(r, 'official_title'))
    df_final['tmdb_ids'] = df_final['verified_recommendations'].apply(lambda r: extract_from_recs(r, 'tmdb_id'))

    # Select and reorder columns for the final output CSV.
    final_columns = ['post_id', 'descriptors', 'image_local_path', 'verified_movies', 'tmdb_ids']
    df_final = df_final[final_columns]

    logging.info(f"Saving final DataFrame to {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Successfully created DataFrame with {len(df_final)} rows. Saved to {OUTPUT_CSV}")
    
    return df_final

if __name__ == "__main__":
    final_df = create_final_dataframe()
    if final_df is not None and not final_df.empty:
        print("\n--- Final DataFrame Head ---")
        print(final_df.head())
