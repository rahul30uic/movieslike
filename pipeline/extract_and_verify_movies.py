import os
import json
import sqlite3
import time
import pandas as pd
import re
import requests
from dotenv import load_dotenv

# Import the new vibe extraction function
from vibe_tag_extractor import extract_vibe_tags

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
INPUT_CSV = os.path.join(DATA_DIR, "Reddit_data.csv")  # Make sure your CSV is in the 'movie extraction' folder
OUTPUT_JSON = os.path.join(DATA_DIR, "phase1_extracted_movies_fresh_run.json")
CACHE_DB = os.path.join(DATA_DIR, "tmdb_cache.sqlite")

# Load environment variables from a local .env file
load_dotenv()

# Make sure these are set in your environment
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY not found in environment variables. Please create a .env file and add it.")

# A reasonable threshold to filter out obscure or incorrect matches
MIN_VOTE_COUNT = 100

# =============================================================================
# Database & API Helpers
# =============================================================================
def setup_database():
    """Initialize SQLite cache to avoid redundant TMDB API calls."""
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tmdb_cache (
            query TEXT PRIMARY KEY,
            response TEXT
        )
    ''')
    conn.commit()
    return conn

def search_tmdb(query, media_type, conn):
    """Search TMDB for a given query and media type, using a cache."""
    cache_key = f"{media_type}:{query}"
    cursor = conn.cursor()
    cursor.execute("SELECT response FROM tmdb_cache WHERE query = ?", (cache_key,))
    result = cursor.fetchone()

    if result:
        return json.loads(result[0])

    time.sleep(0.25)  # Rate limiting
    
    search_url = f"https://api.themoviedb.org/3/search/{media_type}"
    params = {'query': query}
    headers = {'accept': 'application/json'}

    # A v4 access token is a long JWT string containing dots, while a v3 key is shorter.
    if '.' in TMDB_API_KEY:
        # Using v4 Access Token
        headers['Authorization'] = f'Bearer {TMDB_API_KEY}'
    else:
        # Using v3 API Key
        params['api_key'] = TMDB_API_KEY
    
    try:
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        cursor.execute("INSERT OR REPLACE INTO tmdb_cache (query, response) VALUES (?, ?)",
                       (cache_key, json.dumps(data)))
        conn.commit()
        return data
    except requests.exceptions.RequestException as e:
        print(f"      - TMDB API request failed for '{query}': {e}")
        return None

def verify_media_tmdb(query, conn):
    """
    Verify a media title against TMDB, checking both movies and TV shows.
    Returns a dictionary containing the best result and its type ('movie' or 'tv').
    """
    movie_results = search_tmdb(query, 'movie', conn)
    tv_results = search_tmdb(query, 'tv', conn)

    movie_match = movie_results['results'][0] if movie_results and movie_results.get('results') else None
    tv_match = tv_results['results'][0] if tv_results and tv_results.get('results') else None

    best_result = None
    media_type = None

    if movie_match and tv_match:
        if movie_match.get('popularity', 0) >= tv_match.get('popularity', 0):
            best_result = movie_match
            media_type = 'movie'
        else:
            best_result = tv_match
            media_type = 'tv'
    elif movie_match:
        best_result = movie_match
        media_type = 'movie'
    elif tv_match:
        best_result = tv_match
        media_type = 'tv'

    if best_result:
        return {'result': best_result, 'type': media_type}
    return {}

# =============================================================================
# Data Processing & LLM Functions
# =============================================================================
def batch_comments(comments_str):
    """Splits a large string of comments into smaller batches for the LLM."""
    if not isinstance(comments_str, str):
        return []
    comments = comments_str.split("|||")
    batches = []
    current_batch = ""
    for comment in comments:
        if len(current_batch) + len(comment) > 3000: # Keep batches at a reasonable size
            batches.append(current_batch)
            current_batch = comment + "\n"
        else:
            current_batch += comment + "\n"
            
    if current_batch:
        batches.append(current_batch)
    return batches

def clean_title(title):
    """Cleans a potential movie title string."""
    if len(title) > 100: # Likely a sentence, not a title
    if not title or not isinstance(title, str) or len(title.strip()) < 2:
        return ""
        
    title = title.strip()

    # Reject if it's too long or looks like a sentence
    if len(title) > 80 or title.endswith('...'):
        return ""

    # Remove years in parentheses, e.g., (1989) or (2023)
    title = re.sub(r'\s*\(\d{4}\)', '', title).strip()
    # Remove common non-title phrases that the LLM sometimes includes
    non_title_phrases = ["is a movie to slow down", "i second this", "first one i thought of", "this is the one"]

    # Reject if it contains generic "stop words" that are unlikely in a real title
    # The LLM should handle this, but this is a good safeguard.
    stop_words = [
        'a movie', 'the movie', 'a show', 'the show', 'that one', 'like x', 
        'this is', 'i think', 'maybe', 'also', 'the cartoon', 'live action'
    ]
    lower_title = title.lower()
    for phrase in non_title_phrases:
        if phrase in lower_title:
            return "" # Discard if it seems like a comment, not a title
    if any(stop_word in lower_title for stop_word in stop_words):
        return ""
    
    return title

    # If the title is very short after cleaning, it's likely noise
    return title if len(title) > 1 else ""


def extract_titles_with_llm(comments_batch):
    """Prompts a local LLM via Ollama to extract movie/TV titles and returns a list."""
    prompt = f"""
    You are a data extraction assistant. Analyze the following Reddit comments and extract the exact titles of any movies or TV shows mentioned. 
    Output ONLY a JSON array of strings containing the movie titles. 
    If no movies are found, return an empty array: [].
    
    Comments:
    You are a precision data extraction bot. Your task is to analyze the following Reddit comments and extract the exact titles of movies or TV shows.

    RULES:
    1. Extract only specific, real titles (e.g., "Pulp Fiction", "The Matrix", "Stranger Things").
    2. IGNORE generic phrases, placeholders, or descriptions (e.g., "the movie with the guy", "that one show", "movies like X", "the cartoon").
    3. IGNORE names of actors, directors, or streaming services (e.g., "Tom Hanks", "Christopher Nolan", "Netflix", "Hoopla").
    4. Your output MUST be a single, valid JSON object with a single key "titles" containing a list of the extracted title strings.
    5. If no valid titles are found, you MUST return an empty list inside the JSON object.

    EXAMPLE 1:
    Comments: "You should watch The Truman Show. Also, any other movies like it?"
    Output: {{"titles": ["The Truman Show"]}}

    EXAMPLE 2:
    Comments: "I saw a great movie on Hoopla, can't remember the name though."
    Output: {{"titles": []}}

    Comments to analyze:
    ---
    {comments_batch}
    ---
    """
    try:
        payload = {
            "model": "llama3.1",
            "prompt": prompt,
            "format": "json",
            "stream": False
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()

        raw_response_data = response.json()
        llm_response_str = raw_response_data.get("response", "[]")
        llm_response_str = raw_response_data.get("response", "{{}}")
        llm_response_str = raw_response_data.get("response", "{}")
        
        try:
            data = json.loads(llm_response_str)
            extracted_titles = set()

            if isinstance(data, list):
                # Handles a simple JSON array of strings
                for item in data:
                    if isinstance(item, str):
                        extracted_titles.add(item)

            elif isinstance(data, dict):
                # Case 1: Check for a known wrapper key like {"movies": ["Title A", ...]}
                found_wrapper = False
                for key in ["movies", "movie_titles", "titles", "movieTitles"]:
                    if isinstance(data.get(key), list):
                        for item in data[key]:
                            if isinstance(item, str):
                                extracted_titles.add(item)
                        found_wrapper = True
                        break
                
                # Case 2: If no wrapper, assume categorized format like {"Title A": ["Related"], ...}
                if not found_wrapper:
                    for key, value in data.items():
                        if isinstance(key, str):
                            extracted_titles.add(key)
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, str):
                                    extracted_titles.add(item)
            
            if extracted_titles:
            # Expecting a dictionary with a "titles" key.
            if isinstance(data, dict) and 'titles' in data and isinstance(data['titles'], list):
                extracted_titles = {
                    title for item in data['titles'] if isinstance(item, str) and (title := item.strip())
                }
                return list(extracted_titles)
            else:
                # This handles cases where the LLM doesn't follow the format.
                if data and data != {{}}: # Avoids printing debug for valid empty objects
                if data and data != {}: # Avoids printing debug for valid empty objects
                    print(f"  DEBUG: LLM output did not match expected schema {{'titles': [...]}}. Got: {data}")
                return []

            if data and data != []: # Avoids printing debug for valid empty lists
                print(f"  DEBUG: LLM returned valid JSON, but not in a recognized list format: {data}")
            return []

        except json.JSONDecodeError:
            print(f"  DEBUG: Failed to parse LLM response string as JSON. Response was: {llm_response_str}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"LLM API request failed: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
             print(f"  DEBUG: Failing response text from LLM: {response.text}")
        return []

# =============================================================================
# Main Pipeline
# =============================================================================
def run_pipeline():
    """Main function to orchestrate the data extraction and verification pipeline."""
    conn = setup_database()
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    processed_post_ids = set()
    final_results = []

    # Check if a previous run exists and load it to resume.
    if os.path.exists(OUTPUT_JSON):
        print(f"Found existing output file at {OUTPUT_JSON}. Attempting to resume.")
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            final_results = json.load(f)
        for item in final_results:
            processed_post_ids.add(item['post_id'])
        print(f"  -> Resuming with {len(final_results)} posts already processed.")
    else:
        print(f"Starting a fresh run. Output will be saved to {OUTPUT_JSON}")

    for index, row in df.iterrows():
        post_id = str(row.get('id', ''))
        if not post_id or post_id in processed_post_ids:
            continue

        title = row.get('title', '')
        selftext = row.get('selftext', '')
        prompt_text = f"{title}\n{'' if pd.isna(selftext) else selftext}".strip()
        image_url = row.get('post_url', '')
        
        print(f"\nProcessing Post {index + 1}/{len(df)} (ID: {post_id}): {prompt_text[:70]}...")

        # --- New Step: Extract Vibe Tags ---
        print("  -> Extracting vibe tags with LLM...")
        vibe_tags = extract_vibe_tags(title, selftext)
        print(f"  <- Vibe tags extracted: {vibe_tags}")

        comment_batches = batch_comments(row.get('comments', ''))
        if not comment_batches:
            print("  -> No comments found. Skipping.")
            continue

        extracted_titles = set()
        for i, batch in enumerate(comment_batches):
            print(f"  -> Processing comment batch {i+1}/{len(comment_batches)}...")
            llm_titles = extract_titles_with_llm(batch)
            if llm_titles:
                print(f"  <- LLM extracted: {llm_titles}")
                for t in llm_titles:
                    cleaned = clean_title(t)
                    if cleaned:
                        extracted_titles.add(cleaned)
            else:
                print("  <- LLM returned no titles for this batch.")

        if not extracted_titles:
            print("  -> No valid movie/TV titles were extracted for this post.")
            continue

        print(f"  -> Verifying {len(extracted_titles)} unique titles with TMDB...")
        verified_media = []
        verified_media_ids = set()

        for title in sorted(list(extracted_titles)):
            print(f"    - Checking '{title}'...")
            media_info = verify_media_tmdb(title, conn)
            
            if not media_info:
                print(f"      - ❌ Not found on TMDB.")
                continue

            top_match = media_info.get('result')
            media_type = media_info.get('type')
            
            media_id = top_match.get('id')
            vote_count = top_match.get('vote_count', 0)
            
            # --- New Quality Gate ---
            if vote_count < MIN_VOTE_COUNT:
                print(f"      - ⚠️ Low vote count ({vote_count}). Discarding.")
                continue

            if media_type == 'movie':
                official_title = top_match.get('title', 'N/A')
            else: # tv
                official_title = top_match.get('name', 'N/A')

            if media_id in verified_media_ids:
                print(f"      - ℹ️ Already verified '{official_title}' for this post. Skipping.")
                continue

            print(f"      - ✅ Verified: '{official_title}' ({media_type.upper()}, Votes: {vote_count})")
            verified_media.append({
                "tmdb_id": media_id,
                "official_title": official_title,
                "poster_path": top_match.get('poster_path'),
                "vote_count": vote_count
            })
            verified_media_ids.add(media_id)
        
        if verified_media:
            print(f"  -> SUCCESS: Found {len(verified_media)} verified movies/shows for this post.")
            final_results.append({
                "post_id": post_id,
                "prompt_text": prompt_text,
                "image_url": image_url,
                "vibe_tags": vibe_tags,
                "verified_recommendations": verified_media
            })
            processed_post_ids.add(post_id) # Add to set to avoid re-processing duplicates in this run
            # Save after each post to prevent data loss on interruption
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=4, ensure_ascii=False)
        else:
            print("  -> No titles passed the verification filters for this post.")
            
    print(f"\nPipeline complete! Saved {len(final_results)} posts to {OUTPUT_JSON}")
    conn.close()

if __name__ == "__main__":
    run_pipeline()