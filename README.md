# movieslike

Find movies by feeling instead of genre.

There's a subreddit called r/MoviesThatFeelLike where people post a photo or
a mood — a foggy street, a neon diner — and ask "what movies feel like this?"
The comments are full of surprisingly good answers. I scraped about 4,400 of
those posts and turned them into a search engine.

You can describe a mood in words, upload any image that feels right, or just
react to pairs of images and let the app figure out your headspace. It gives
you five movies, not five hundred. That's the point — I built this because I
was tired of scrolling Netflix for 40 minutes and watching nothing.

## how it works

Every post gets a "vibe fingerprint": a vector made of two halves.

- **words half:** Gemini looks at the post (image + title + mood tags) and
  writes a short paragraph describing the vibe. A sentence-embedding model
  (bge) turns that paragraph into numbers.
- **picture half:** SigLIP embeds the images directly.

The two halves get concatenated. Then every *movie* gets its own fingerprint:
the weighted average of all the posts where people recommended it (a post
that shotguns 50 movie titles counts less than one that gives 3). Search is
just cosine similarity against ~17k movie vectors, plus a penalty on movies
that Reddit recommends for literally everything (looking at you, Blade
Runner), and a floor on TMDB vote count so you don't get movies nobody can
find.

## how I know it works

There are no labels for "these two posts have the same vibe", but there's a
trick: if two posts' commenters recommended the same *obscure* movie, the
posts almost certainly share a vibe. So the eval is: take a post, find its
nearest neighbors in embedding space, check how often they share rare movies,
and compare against random pairs.

That number caught real problems. The first version averaged text and image
embeddings 50/50, and the eval showed the text side was scoring *below
random* — SigLIP's text encoder just can't handle bags of mood words, and
averaging was dragging the good image vectors down with it. Every version
since had to beat the previous number:

| version | lift over random | 95% CI |
|---|---|---|
| siglip, 50/50 text+image average | 5.0x | |
| siglip, fusion weight swept | 6.6x | |
| gemini captions alone | 4.8x | |
| hybrid (captions + images) | 7.5x | [7.3, 7.8] |
| hybrid + LoRA fine-tune (what's live now) | **8.1x** | [7.8, 8.4] |

The LoRA row: I fine-tuned SigLIP's vision tower (rank 8, only 295k params)
using the corpus's own signal as supervision — images from posts that share a
rare movie are positive pairs. On held-out posts it improved image retrieval
by 18.7%.

Shipping it was its own lesson. Dropping the tuned model into the pipeline
actually made end-to-end retrieval *worse* at first — contrastive training
changes the spread of the similarity scores, which broke the old mixing
weight between the two halves. Re-tuning that one number fixed it and then
some. Offline wins don't automatically transfer; measure before you ship.

There's also a small human eval (25 mood scenarios, rated by hand) in
`eval/golden_set.py`, because the rare-movie-overlap metric is a proxy and I
wanted an independent check on it.

Sanity check that the space is real — nearest movies by vibe:

- *Lost in Translation* → Chungking Express, Her, Fallen Angels
- *The Thing* → Jacob's Ladder, In the Mouth of Madness, The Void
- *Fargo* → Winter's Bone, True Detective, Prisoners

None of that follows genre tags. "Neon urban loneliness" is not a TMDB
category.

## the whole thing runs in your browser

There's no backend. transformers.js runs the models client-side: bge for
text, and my LoRA-tuned SigLIP exported to quantized ONNX (94MB, cosine
0.984 vs the pytorch version) for images. The movie index is a static 29MB
file. First visit downloads the models, after that it's all local. Free to
host, nothing to keep alive, and the fine-tune is literally what embeds your
uploads.

Some extras I built because I wanted to see inside the model:

- `/atlas` — a UMAP map of the whole embedding space, drawn with the actual
  corpus images. The foggy-coast region and the neon-city region are real
  neighborhoods the model found on its own.
- upload an image and you get a heatmap showing *which parts of your image*
  drove the top match (SigLIP patch tokens scored against the matched
  movie's vector).
- the probe (the landing page) is a little Bayesian machine: each pick
  between two images reweights a posterior over 500 mood particles, pairs
  are chosen to split the remaining uncertainty, and you can watch the
  "mood lock" meter fill as it converges. There's also a diffusion-generated
  image pool as a rights-clean alternative to the Reddit images.
- it remembers your taste across sessions (locally, resettable) and blends
  it lightly into rankings.

## repo layout

| folder | what |
|---|---|
| `pipeline/` | scraping → verification → captions → embeddings → indexes |
| `eval/` | the eval harness + every result as json |
| `frontend/` | next.js app, browser inference in `lib/engine.js` |
| `api/` | fastapi server (only needed for local dev now) |
| `data/` | gitignored; everything regenerable via pipeline |

## running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # TMDB + Gemini keys in .env for the pipeline

# pipeline, in order (each step caches)
python pipeline/extract_and_verify_movies.py
python pipeline/extract_vibe_tags_from_reddit.py
python pipeline/generate_vibe_captions.py        # ~$5 of Gemini
python pipeline/encode_modalities.py
python pipeline/embed_captions.py
python pipeline/train_siglip_lora.py             # optional but worth it
python pipeline/ship_lora.py
python pipeline/build_hybrid_vectors.py --image-npz data/post_modality_vectors_lora.npz
python pipeline/build_movie_vectors.py --posts-file data/posts_with_hybrid_vectors.json --output data/movie_vectors_hybrid.json
python pipeline/export_web_index.py

# frontend
cd frontend && npm i && NEXT_PUBLIC_USE_BROWSER_ENGINE=true npm run dev
```

## known problems

- the corpus skews moody/atmospheric. it's deep on "foggy seaside dread",
  thin on "fun date-night comedy". about 45% of movies have only one
  supporting post, so their vectors are noisy.
- TMDB title matching has some wrong matches I haven't cleaned up yet (a
  1978 thriller got matched to a 2019 docuseries in ~200 posts).
- the training signal and the eval metric come from the same source
  (co-recommendation), which is why the human golden set exists — the split
  rules out memorization but not the signal's own biases.
- text-only posts (~8% of corpus) still retrieve badly. known, unfixed.

Movie data from [TMDB](https://www.themoviedb.org). This product uses the
TMDB APIs but is not endorsed or certified by TMDB. Mood images come from
public Reddit posts; rights remain with their owners — contact for removal.
