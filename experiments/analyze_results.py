import json
import os
from collections import Counter
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "phase1_extracted_movies.json")

def analyze_data():
    """
    Analyzes the extracted movie data JSON file and prints statistics.
    """
    if not os.path.exists(INPUT_JSON):
        print(f"Error: Input file not found at {INPUT_JSON}")
        print("Please make sure the JSON file from the extraction script is present.")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_posts = len(data)
    all_recommendations = []
    unique_media_ids = set()

    for post in data:
        recommendations = post.get("verified_recommendations", [])
        # Deduplicate recommendations within the same post before adding to the main list
        post_unique_recs = {rec['tmdb_id']: rec for rec in recommendations}.values()
        for rec in post_unique_recs:
            all_recommendations.append(rec)
            unique_media_ids.add(rec['tmdb_id'])

    total_unique_movies = len(unique_media_ids)
    total_recommendations_after_post_dedup = len(all_recommendations)

    print("="*50)
    print("      Dataset Statistics")
    print("="*50)
    print(f"Total Posts Analyzed: {total_posts}")
    print(f"Total Recommendations (post-deduplication): {total_recommendations_after_post_dedup}")
    print(f"Total Unique Movies/Shows: {total_unique_movies}")
    print("-"*50)

    if not all_recommendations:
        print("No recommendations found to analyze further.")
        return

    # Most recommended movies
    recommendation_counts = Counter(rec['tmdb_id'] for rec in all_recommendations)
    most_common = recommendation_counts.most_common(15)

    print("\nTop 15 Most Recommended Movies/Shows:")
    title_map = {rec['tmdb_id']: rec['official_title'] for rec in all_recommendations}
    for tmdb_id, count in most_common:
        title = title_map.get(tmdb_id, f"ID: {tmdb_id}")
        print(f"  - \"{title}\": {count} times")

    # Vote count analysis
    vote_counts = [rec['vote_count'] for rec in all_recommendations if 'vote_count' in rec]
    if vote_counts:
        df_votes = pd.Series(vote_counts)
        print("\nVote Count Statistics:")
        print(f"  - Average Vote Count: {df_votes.mean():.2f}")
        print(f"  - Median Vote Count: {df_votes.median()}")
        print(f"  - Movies with > 10,000 votes: {len([v for v in vote_counts if v > 10000])}")
        print(f"  - Movies with < 100 votes: {len([v for v in vote_counts if v < 100])}")

    # =========================================================================
    # Per-Post Analysis Table
    # =========================================================================
    print("\n" + "="*50)
    print("           Movies Recommended Per Post")
    print("="*50)

    post_recommendations_data = []
    for post in data:
        post_id = post.get('post_id')
        prompt_text = post.get('prompt_text', '').strip().replace('\n', ' ')
        recommendations = post.get("verified_recommendations", [])

        # Deduplicate recommendations based on tmdb_id
        unique_recs = {rec['tmdb_id']: rec for rec in recommendations}.values()
        
        # Get the official titles of the unique recommendations, sorted alphabetically
        movie_titles = sorted([rec['official_title'] for rec in unique_recs])

        post_recommendations_data.append({
            "Post ID": post_id,
            "Prompt": prompt_text[:80] + '...' if len(prompt_text) > 80 else prompt_text,
            "Unique Recommendations": ", ".join(movie_titles)
        })

    if post_recommendations_data:
        df_posts = pd.DataFrame(post_recommendations_data)
        print(df_posts.to_string())

    print("\n" + "="*50)

if __name__ == "__main__":
    analyze_data()