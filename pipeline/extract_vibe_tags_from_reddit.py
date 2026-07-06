import os
import json
import pandas as pd
from tqdm import tqdm

# Import the new vibe extraction function
from vibe_tag_extractor import extract_vibe_tags

import os
import json
import pandas as pd
from tqdm import tqdm

# Assuming vibe_tag_extractor.py is in the same directory and has logging configured
from vibe_tag_extractor import extract_vibe_tags, logging

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
INPUT_CSV = os.path.join(DATA_DIR, "Reddit_data.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "phase2_vibe_tags_output.json")

# =============================================================================
# Main Pipeline
# =============================================================================
def run_vibe_extraction():
    """
    Reads Reddit data, extracts descriptive vibe tags from each post's title
    and selftext using an LLM, and saves the results.
    """
    logging.info(f"Loading Reddit posts from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        logging.error(f"Input file not found at {INPUT_CSV}. Please ensure the file exists.")
        return

    processed_post_ids = set()
    all_results = []

    # Resume functionality: Load existing data if output file is present
    if os.path.exists(OUTPUT_JSON):
        logging.info(f"Resuming from existing output file: {OUTPUT_JSON}")
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            processed_post_ids = {item['post_id'] for item in all_results if 'post_id' in item}
            logging.info(f"Loaded {len(all_results)} previously processed posts.")
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"Could not parse existing output file. Starting fresh. Error: {e}")
            all_results = []
    else:
        logging.info("No existing output file found. Starting a fresh run.")

    # Use tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Vibe Tags"):
        post_id = str(row.get('id', ''))

        if not post_id or post_id in processed_post_ids:
            continue

        title = row.get('title', '')
        selftext = row.get('selftext', '')

        if pd.isna(title) and pd.isna(selftext):
            continue
            
        vibe_tags = extract_vibe_tags(title, selftext)

        if vibe_tags:
            all_results.append({"post_id": post_id, "title": title, "selftext": selftext, "vibe_tags": vibe_tags})
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)

    logging.info(f"\nPipeline complete. Total posts with vibe tags: {len(all_results)}")
    logging.info(f"Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    run_vibe_extraction()
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
INPUT_CSV = os.path.join(DATA_DIR, "Reddit_data.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "phase2_vibe_tags_output.json")

# =============================================================================
# Main Pipeline
# =============================================================================
def run_pipeline():
    """
    Main function to orchestrate the vibe tag extraction pipeline.
    It reads posts from a CSV, extracts vibe tags from the title and selftext,
    and saves the results to a JSON file, resuming if the output file exists.
    """
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    processed_post_ids = set()
    final_results = []

    if os.path.exists(OUTPUT_JSON):
        print(f"Found existing output file at {OUTPUT_JSON}. Attempting to resume.")
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            final_results = json.load(f)
        for item in final_results:
            if 'post_id' in item:
                processed_post_ids.add(item['post_id'])
        print(f"  -> Resuming with {len(final_results)} posts already processed.")
    else:
        print(f"Starting a fresh run. Output will be saved to {OUTPUT_JSON}")

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Posts"):
        post_id = str(row.get('id', ''))
        if not post_id or post_id in processed_post_ids:
            continue

        title = row.get('title', '')
        selftext = row.get('selftext', '')
        
        # Skip posts with no text content to analyze
        if pd.isna(title) and pd.isna(selftext):
            continue

        vibe_tags = extract_vibe_tags(title, selftext)
        
        if not vibe_tags:
            # Optional: log or print that no tags were found for this post
            continue
        
        final_results.append({
            "post_id": post_id,
            "title": title,
            "selftext": selftext,
            "vibe_tags": vibe_tags
        })
        processed_post_ids.add(post_id)
        
        # Save after each successful extraction to prevent data loss
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print(f"\nPipeline complete! Saved {len(final_results)} posts to {OUTPUT_JSON}")

if __name__ == "__main__":
    run_pipeline()
