"""
Generate a unified "vibe caption" per post with Gemini (vision + text).

For each post we send the images (up to 3, downscaled), the Reddit title, and
the extracted descriptors, and ask for one evocative 40-80 word description of
the vibe. The caption is later embedded with a strong text-only model — this
replaces vector-level modality fusion with fusion in language.

Resumable: already-captioned post_ids in the output file are skipped.

Usage:
    python generate_vibe_captions.py --limit 50     # sample run
    python generate_vibe_captions.py                # full corpus
"""

import argparse
import ast
import io
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PIL import Image
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
INPUT_CSV = os.path.join(DATA_DIR, "final_dataset.csv")
REDDIT_CSV = os.path.join(DATA_DIR, "Reddit_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "vibe_captions.jsonl")

MODEL = "gemini-2.5-flash"
MAX_IMAGES_PER_POST = 3
MAX_IMAGE_DIM = 640
CONCURRENCY = 8
MAX_RETRIES = 6
DEFAULT_RPM = 5  # free-tier gemini-2.5-flash limit


class RateLimiter:
    """Global pacer: at most `rpm` request starts per minute across threads."""

    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.next_slot = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            slot = max(self.next_slot, now)
            self.next_slot = slot + self.interval
        time.sleep(max(0.0, slot - time.monotonic()))

PROMPT = """You are helping build a movie-vibe search engine. The input is a Reddit post \
from a community where people share images and moods and ask for movies that *feel* like that.

Write a single evocative description (40-80 words) of the VIBE this post expresses. Cover, \
as applicable: atmosphere and mood, color palette and light, setting and texture, pacing and \
energy, emotional undertone. Use concrete sensory language. Synthesize the images and the \
words into ONE coherent vibe — if they tension against each other (e.g. cheerful imagery, \
unsettling words), describe that tension explicitly.

Do NOT mention: movie titles, Reddit, the poster, or this request. Output only the description."""


def safe_list(val):
    if pd.isna(val):
        return []
    try:
        v = ast.literal_eval(val)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


def load_env_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(SCRIPT_DIR), ".env")
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GEMINI_API_KEY not found in environment or .env")


def load_posts():
    df = pd.read_csv(INPUT_CSV)
    df["descriptors"] = df["descriptors"].apply(safe_list)

    titles = pd.read_csv(REDDIT_CSV, usecols=["id", "title"])
    title_map = dict(zip(titles["id"], titles["title"]))

    posts = []
    for _, row in df.iterrows():
        paths = []
        if isinstance(row["image_local_path"], str):
            for p in row["image_local_path"].split("|"):
                ap = os.path.join(DATA_DIR, p.strip())
                if os.path.exists(ap):
                    paths.append(ap)
        posts.append({
            "post_id": row["post_id"],
            "title": title_map.get(row["post_id"], ""),
            "descriptors": row["descriptors"],
            "image_paths": paths[:MAX_IMAGES_PER_POST],
        })
    return posts


def shrink(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))
    return img


def caption_post(client, post, limiter, model):
    parts = [PROMPT]
    context = []
    if post["title"]:
        context.append(f'Post title: "{post["title"]}"')
    if post["descriptors"]:
        context.append(f"Extracted vibe descriptors: {', '.join(post['descriptors'])}")
    if not context and not post["image_paths"]:
        return None  # nothing to describe
    parts.append("\n".join(context) if context else "(no text — describe the images alone)")
    for p in post["image_paths"]:
        try:
            parts.append(shrink(p))
        except Exception as e:
            logging.warning(f"{post['post_id']}: unreadable image {p}: {e}")

    delay = 20.0
    for attempt in range(MAX_RETRIES):
        try:
            limiter.wait()
            resp = client.models.generate_content(
                model=model,
                contents=parts,
                # No hidden "thinking" tokens — a 60-word caption doesn't need
                # them and they multiply the per-call cost.
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            text = (resp.text or "").strip()
            if text:
                return text
            raise RuntimeError("empty response")
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logging.error(f"{post['post_id']}: giving up: {e}")
                return None
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(delay)
                delay = min(delay * 2, 120)
            else:
                time.sleep(2 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Only caption the first N pending posts.")
    ap.add_argument("--rpm", type=int, default=DEFAULT_RPM, help="Requests per minute budget.")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["post_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    posts = [p for p in load_posts() if p["post_id"] not in done]
    if args.limit:
        posts = posts[:args.limit]
    logging.info(f"{len(done)} already captioned; {len(posts)} to go.")
    if not posts:
        return

    # 90s hard timeout per request — without it, one hung connection per worker
    # thread silently stalls the whole run.
    client = genai.Client(api_key=load_env_key(), http_options={"timeout": 90_000})
    limiter = RateLimiter(args.rpm)
    lock = threading.Lock()
    n_ok = 0

    workers = min(CONCURRENCY, max(1, args.rpm // 2))
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(caption_post, client, p, limiter, args.model): p for p in posts}
            for i, fut in enumerate(as_completed(futures), 1):
                post = futures[fut]
                caption = fut.result()
                if caption:
                    with lock:
                        out.write(json.dumps({"post_id": post["post_id"], "caption": caption}) + "\n")
                        out.flush()
                        n_ok += 1
                if i % 50 == 0:
                    logging.info(f"{i}/{len(posts)} processed ({n_ok} ok)")

    logging.info(f"Done: {n_ok}/{len(posts)} captioned. Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
