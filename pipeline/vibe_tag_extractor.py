import os
import json
import logging
from typing import List, Dict, Any

import requests
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

# Configure logging to provide clear output for warnings and errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# API endpoint can be easily swapped here for cloud providers like Groq, Together, etc.
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.1"
API_TIMEOUT_SECONDS = 45

# =============================================================================
# LLM Prompting
# =============================================================================

SYSTEM_PROMPT = """
You are a highly specialized cinematic analyst AI. Your sole purpose is to extract atmospheric, visual, and thematic keywords or 'vibe tags' and 'mood tags' from unstructured text. You must ignore all conversational filler, questions, and pleasantries. Your output must be a single, valid JSON object.
"""

PROMPT_TEMPLATE = """
Analyze the text below. Extract a list of descriptive keywords and short phrases that capture its cinematic mood, atmosphere, visual style, or core themes.

- Focus on rich, evocative terms (e.g., "sun-drenched nostalgia", "gritty urban decay", "dreamlike surrealism", "cozy autumn evening").
- Exclude generic, conversational, or non-descriptive text (e.g., "movies that feel like this", "any recommendations?", "thanks in advance", "I've already seen...").
- If the text contains no descriptive information, you MUST return an empty list.

Return ONLY a JSON object with the key "vibe_tags" containing a list of strings. Do not include any other text, explanations, or markdown.

Example Input: "I'm looking for a movie that feels like a warm, foggy seaside town with some really bizarre and complex characters. Thanks!"
Example Output:
{{"vibe_tags": ["warm", "foggy", "seaside town", "bizarre characters", "complex characters"]}}

Example Input: "Any suggestions?"
Example Output:
{{"vibe_tags": []}}

Text to Analyze:
---
{text_input}
---
"""

# =============================================================================
# Core Logic
# =============================================================================

def _call_llm_api(prompt: str) -> Dict[str, Any]:
    """
    Internal function to make the API call to the LLM endpoint.
    This is modularized to allow easy swapping of API providers in the future.
    
    Raises:
        requests.exceptions.RequestException: If the API call fails.
    """
    payload = {
        "model": LLM_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }
    response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
    return response.json()

def extract_vibe_tags(title: str, body: str) -> List[str]:
    """
    Extracts cinematic vibe tags from a post's title and body using an LLM.

    This function constructs a prompt, sends it to a Llama 3.1 endpoint,
    and safely parses the JSON response to extract a list of descriptive tags.
    It handles API errors, JSON decoding errors, and schema validation gracefully.

    Args:
        title: The title of the post.
        body: The body content of the post.

    Returns:
        A list of string tags, or an empty list if no tags are found or an error occurs.
    """
    if not title and not body:
        return []

    combined_text = f"Title: {title}\n\nBody: {body}".strip()
    prompt = PROMPT_TEMPLATE.format(text_input=combined_text)
    response_str = ""

    try:
        api_response = _call_llm_api(prompt)
        
        # Ollama with format=json wraps the model's JSON string in its own response object.
        response_str = api_response.get("response")
        if not response_str:
            logging.warning("LLM response was empty or missing the 'response' key.")
            return []

        data = json.loads(response_str)

        vibe_tags = data.get("vibe_tags")
        if vibe_tags is None or not isinstance(vibe_tags, list):
            logging.warning(f"LLM output did not match expected schema. Got: {data}")
            return []
        
        # Ensure all items in the list are strings before returning
        return [str(tag) for tag in vibe_tags if isinstance(tag, str)]

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return []
    except json.JSONDecodeError:
        logging.warning(f"Failed to decode JSON from LLM response string: {response_str}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred in extract_vibe_tags: {e}")
        return []

# =============================================================================
# Batch Processing Example
# =============================================================================

def batch_process_posts(posts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Processes a list of posts to extract vibe tags, showing a progress bar."""
    results = []
    for post in tqdm(posts, desc="Extracting Vibe Tags"):
        # Use 'selftext' as the key for the body, matching the Reddit dataset
        tags = extract_vibe_tags(post.get("title", ""), post.get("selftext", ""))
        results.append({
            "post_id": post.get("id"),
            "original_title": post.get("title", ""),
            "vibe_tags": tags
        })
    return results

if __name__ == "__main__":
    # Sample data mimicking rows from your Reddit_data.csv
    sample_posts = [
        {"id": "sample1", "title": "Movies that feel like this?", "selftext": "I'm looking for a movie that feels like a warm, foggy seaside town with some really bizarre and complex characters. Thanks!"},
        {"id": "sample2", "title": "Help me find a film", "selftext": "I saw it a while ago, can't remember the name."},
        {"id": "sample3", "title": "Need something with gritty, neon-soaked, urban decay vibes.", "selftext": "Think Blade Runner but maybe more modern. Something with a lonely protagonist."},
        {"id": "sample4", "title": "Any suggestions?", "selftext": ""},
        {"id": "sample5", "title": "Cozy autumn evening movie", "selftext": "I want something that feels like putting on a warm sweater, drinking tea, and watching the rain outside. Maybe a little bit of mystery but not scary."}
    ]

    logging.info("Starting batch processing of sample posts...")
    processed_results = batch_process_posts(sample_posts)
    logging.info("Batch processing complete.")

    # Pretty-print the results
    for result in processed_results:
        print(f"\nPost ID: {result['post_id']} (Title: {result['original_title']})")
        print(f"  -> Extracted Tags: {result['vibe_tags']}")
