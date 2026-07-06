import pandas as pd
import numpy as np
import os
import logging
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
from io import BytesIO
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
# --- Dependencies ---
# You'll need to install the following packages:
# pip install pandas numpy umap-learn matplotlib seaborn Pillow tqdm

# --- File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Input from the embedding/clustering script
VECTORS_INPUT_JSONL = os.path.join(SCRIPT_DIR, "posts_with_vectors.json")
# Outputs for this script
PLOT_OUTPUT_PNG = os.path.join(SCRIPT_DIR, "cluster_visualization.png")
REPORT_OUTPUT_HTML = os.path.join(SCRIPT_DIR, "cluster_image_report.html")

# --- Visualization Parameters ---
# Max images to show per cluster in the HTML report to keep file size reasonable
MAX_IMAGES_PER_CLUSTER = 25
THUMBNAIL_SIZE = (150, 150)

# --- UMAP Parameters ---
# These can be tuned, but are good defaults.
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# Helper Functions
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Loads the clustered posts data from a JSONL file."""
    logging.info(f"Loading clustered data from {filepath}...")
    if not os.path.exists(filepath):
        logging.error(f"Input file not found: {filepath}")
        logging.error("Please run 'generate_embeddings_and_clusters.py' first to create this file.")
        return None
    try:
        df = pd.read_json(filepath, lines=True)
        logging.info(f"Successfully loaded {len(df)} posts with vectors.")
        return df
    except Exception as e:
        logging.error(f"Failed to load or parse {filepath}: {e}")
        return None

def image_to_base64(img_path: str) -> str:
    """Converts an image file to a base64 encoded string for HTML embedding."""
    try:
        with Image.open(img_path) as img:
            img.thumbnail(THUMBNAIL_SIZE)
            buffered = BytesIO()
            # Ensure image is RGB before saving as JPEG
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logging.warning(f"Could not process image {img_path}: {e}")
        return ""

# =============================================================================
# Core Visualization Logic
# =============================================================================

def create_2d_scatterplot(df: pd.DataFrame, vectors: np.ndarray):
    """
    Reduces vector dimensions using UMAP and creates a 2D scatterplot of the clusters.
    """
    logging.info(f"Performing UMAP dimensionality reduction on {len(vectors)} vectors...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        random_state=UMAP_RANDOM_STATE,
        metric='cosine'
    )
    embedding_2d = reducer.fit_transform(vectors)

    plot_df = pd.DataFrame(embedding_2d, columns=['x', 'y'])
    plot_df['cluster'] = df['cluster_id'].astype(str) # Use string for categorical plotting

    logging.info("Generating scatterplot...")
    plt.figure(figsize=(16, 12))
    
    # Separate outliers from clustered points for different styling
    outliers = plot_df[plot_df['cluster'] == '-1']
    clustered_points = plot_df[plot_df['cluster'] != '-1']
    
    num_clusters = len(clustered_points['cluster'].unique())
    palette = sns.color_palette("hsv", num_clusters)

    # Plot the main clusters
    ax = sns.scatterplot(
        data=clustered_points,
        x='x',
        y='y',
        hue='cluster',
        palette=palette,
        s=50,
        alpha=0.7,
        legend='full'
    )

    # Plot outliers with a distinct style
    if not outliers.empty:
        sns.scatterplot(
            data=outliers,
            x='x',
            y='y',
            color='gray',
            s=10,
            alpha=0.2,
            label='Outliers (-1)',
            ax=ax
        )

    ax.set_title('2D UMAP Projection of Post Clusters', fontsize=18)
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    logging.info(f"Saving plot to {PLOT_OUTPUT_PNG}...")
    plt.savefig(PLOT_OUTPUT_PNG, dpi=300)
    plt.close()

def generate_html_report(df: pd.DataFrame):
    """
    Generates a self-contained HTML file with image grids for each cluster.
    """
    logging.info("Generating HTML image report...")
    
    # The root directory for images is the same as the script's directory
    image_root = SCRIPT_DIR

    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cluster Image Report</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f9; }
            h1 { text-align: center; color: #333; }
            h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-top: 40px; }
            .cluster-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px; }
            .post-item { background: #fff; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }
            .post-item img { max-width: 100%; height: auto; display: block; }
            .post-item p { font-size: 12px; color: #666; padding: 5px; margin: 0; word-wrap: break-word; }
        </style>
    </head>
    <body>
        <h1>Cluster Image Report</h1>
    """

    # Group by cluster_id and sort to have outliers (-1) at the end
    grouped = df.groupby('cluster_id')
    sorted_clusters = sorted(grouped.groups.keys(), key=lambda x: (x == -1, x))

    for cluster_id in tqdm(sorted_clusters, desc="Building HTML Report"):
        cluster_df = grouped.get_group(cluster_id)
        cluster_name = "Outliers" if cluster_id == -1 else f"Cluster {cluster_id}"
        html += f"<h2>{cluster_name} ({len(cluster_df)} posts)</h2>\n"
        html += '<div class="cluster-grid">\n'

        # Limit the number of images displayed per cluster
        for _, row in cluster_df.head(MAX_IMAGES_PER_CLUSTER).iterrows():
            image_path_str = row.get('image_local_path')
            
            if pd.isna(image_path_str):
                continue

            # Take the first image from a potentially pipe-separated list
            first_image_rel_path = image_path_str.split('|')[0].strip()
            full_img_path = os.path.join(image_root, first_image_rel_path)

            if os.path.exists(full_img_path):
                base64_img = image_to_base64(full_img_path)
                if base64_img:
                    # Use descriptors as a tooltip
                    descriptors = ', '.join(row.get('descriptors', []))
                    html += f'<div class="post-item" title="Post ID: {row["post_id"]}\\nDescriptors: {descriptors}">'
                    html += f'<img src="{base64_img}" alt="Post ID {row["post_id"]}">'
                    html += f'<p>ID: {row["post_id"]}</p>'
                    html += '</div>\n'
        
        html += '</div>\n'

    html += """
    </body>
    </html>
    """

    logging.info(f"Saving HTML report to {REPORT_OUTPUT_HTML}...")
    with open(REPORT_OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to orchestrate the visualization pipeline."""
    df = load_data(VECTORS_INPUT_JSONL)
    if df is None or df.empty:
        return

    # Extract vectors into a NumPy array for UMAP
    vectors = np.array(df['combined_vector'].tolist())
    
    # 1. Create and save the 2D scatter plot
    create_2d_scatterplot(df, vectors)
    
    # 2. Create and save the detailed HTML image report
    generate_html_report(df)

    logging.info("Visualization pipeline finished successfully.")
    logging.info(f"Plot saved to: {PLOT_OUTPUT_PNG}")
    logging.info(f"Image report saved to: {REPORT_OUTPUT_HTML}")

if __name__ == "__main__":
    main()
