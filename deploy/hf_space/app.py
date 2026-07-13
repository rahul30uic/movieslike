"""
Hugging Face Space entrypoint (Gradio SDK, free tier).

Mounts the full Movieslike FastAPI (main.py, copied alongside) into a Gradio
app: the Space's page shows a small built-in demo UI, while the JSON API
(/recommendations, /recommendations/text, /recommendations/image) stays
available at the same host for the Vercel frontend.
"""

import os
import sys

import gradio as gr
import numpy as np
import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# In the Space, artifacts live at <repo>/data (main.py's default assumes ../data)
os.environ.setdefault("MOVIESLIKE_DATA_DIR", os.path.join(HERE, "data"))
import main  # noqa: E402  (the FastAPI app + retrieval logic)

POSTER = "https://image.tmdb.org/t/p/w342"


def _gallery(recs):
    out = []
    for m in recs:
        url = f"{POSTER}{m.poster_path}" if m.poster_path else None
        out.append((url, f"{m.title}  ·  {m.n_posts} posts"))
    return out


def text_search(query, alpha, min_votes):
    if main.embedder is None:
        raise gr.Error("Models are still loading — try again in ~30s.")
    if not query or len(query.strip()) < 2:
        raise gr.Error("Describe the vibe in a few words.")
    q = main.embedder.encode(main.QUERY_PREFIX + query.strip(), normalize_embeddings=True)
    target = np.concatenate([q, np.zeros_like(q)]).astype(np.float32)
    return _gallery(main.rank(target, alpha, 10, main.DEFAULT_MIN_SUPPORT, int(min_votes)))


def image_search(img, alpha, min_votes):
    if main.image_encoder is None:
        raise gr.Error("Models are still loading — try again in ~30s.")
    if img is None:
        raise gr.Error("Upload an image first.")
    img.thumbnail((512, 512))
    v = main.image_encoder.encode(img, convert_to_numpy=True)
    v = v / (np.linalg.norm(v) + 1e-9)
    target = np.concatenate([np.zeros_like(v), v]).astype(np.float32)
    return _gallery(main.rank(target, alpha, 10, main.DEFAULT_MIN_SUPPORT, int(min_votes)))


with gr.Blocks(title="Movieslike") as demo:
    gr.Markdown(
        "# 🎬 Movieslike — vibe-based movie search\n"
        "Built from 4,400 Reddit posts of *\"movies that feel like this?\"* — "
        "[code & write-up](https://github.com/rahul30uic/movieslike). "
        "Describe a mood, or upload an image that feels right."
    )
    with gr.Row():
        alpha = gr.Slider(0, 1, value=0.3, label="Hidden gems ← → dial (alpha)")
        min_votes = gr.Slider(0, 5000, value=500, step=50, label="Recognizability floor (TMDB votes)")
    with gr.Tab("Describe the vibe"):
        q = gr.Textbox(label="", placeholder='e.g. "cozy rainy night, gentle loneliness"')
        btn_t = gr.Button("Find movies", variant="primary")
    with gr.Tab("Upload a mood image"):
        img = gr.Image(type="pil", label="")
        btn_i = gr.Button("Match this image", variant="primary")
    gallery = gr.Gallery(label="Your movies", columns=5, height="auto")

    btn_t.click(text_search, [q, alpha, min_votes], gallery)
    q.submit(text_search, [q, alpha, min_votes], gallery)
    btn_i.click(image_search, [img, alpha, min_votes], gallery)

app = gr.mount_gradio_app(main.app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
