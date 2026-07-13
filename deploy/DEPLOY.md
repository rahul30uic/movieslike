# Deploying the live demo

Two free-tier pieces: the API on a Hugging Face Space (Docker), the frontend
on Vercel. Total cost: $0. The public demo serves the anchor grid, text
search, and image upload; the probe stays local-only (its 3.5GB of
Reddit-sourced images shouldn't be redistributed).

## 1. API → Hugging Face Space

1. Create a (free) account at huggingface.co, then **New Space** →
   SDK: **Docker** → name it e.g. `movieslike-api` → CPU basic (free).
2. From the repo root:

   ```bash
   git clone https://huggingface.co/spaces/<YOUR_HF_USER>/movieslike-api hf-space
   cp deploy/Dockerfile hf-space/Dockerfile
   cp -r api hf-space/api
   cp -r deploy/data hf-space/deploy-data && mv hf-space/deploy-data hf-space/data
   # Dockerfile copies api/ and data/ — adjust COPY paths to match:
   #   COPY api/ /app/api/     COPY data/ /app/data/
   cd hf-space && git add -A && git commit -m "deploy api" && git push
   ```

3. In the Space's **Settings → Variables**, set
   `ALLOWED_ORIGINS=https://<your-vercel-domain>` once you know it (step 2.3).
4. The Space builds (~10 min, model downloads baked into the image) and serves
   at `https://<YOUR_HF_USER>-movieslike-api.hf.space`.

## 2. Frontend → Vercel

1. vercel.com → **Add New Project** → import the `movieslike` GitHub repo →
   set **Root Directory** to `frontend/`.
2. Environment variables:
   - `NEXT_PUBLIC_API_URL` = `https://<YOUR_HF_USER>-movieslike-api.hf.space`
   - `NEXT_PUBLIC_ENABLE_PROBE` = `false`  (probe is local-only)
3. Deploy. Note the domain Vercel assigns, and put it into the Space's
   `ALLOWED_ORIGINS` (step 1.3), comma-separated if you add a custom domain.

## Gotchas

- Free HF Spaces sleep after ~48h idle; first request after sleep takes
  ~1 min. Before an interview round, open the link once to warm it.
- The frontend's `public/reddit_images_*` symlink resolves to nothing in
  Vercel's build — that's fine, nothing in the deployed pages references it.
- To update the live API after pipeline changes: re-run
  `python pipeline/export_deploy_artifacts.py`, copy `deploy/data` into the
  Space checkout, commit, push.
