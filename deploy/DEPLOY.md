# Deploying the live demo

Two free-tier pieces: the API on a Hugging Face Space (**Gradio SDK** — the
Docker SDK went paid-only in 2026), the frontend on Vercel. Total cost: $0.

The Space runs `deploy/hf_space/app.py`, which mounts the full FastAPI inside
a Gradio app — so the Space page doubles as a self-contained demo UI, while
the JSON API stays available for the Vercel frontend at the same host.

The public demo serves the anchor grid, text search, and image upload; the
probe stays local-only (its 3.5GB of Reddit-sourced images shouldn't be
redistributed — the API detects the missing data and disables /probe).

## 1. API → Hugging Face Space (Gradio SDK)

1. Create a free account at huggingface.co → **New Space**:
   SDK **Gradio** (blank template) · CPU basic (free) · name `movieslike-api`.
2. Get a **Write** token (Settings → Access Tokens) and run
   `huggingface-cli login` locally.
3. From the repo root, assemble and push the Space:

   ```bash
   git clone https://huggingface.co/spaces/<HF_USER>/movieslike-api hf-space
   cp deploy/hf_space/{app.py,requirements.txt,README.md} hf-space/
   cp api/main.py hf-space/
   mkdir -p hf-space/data && cp deploy/data/* hf-space/data/
   cd hf-space && git add -A && git commit -m "deploy" && git push
   ```

4. First build takes a few minutes (model downloads happen on first boot).
   The Space serves at `https://<HF_USER>-movieslike-api.hf.space`.
5. Space **Settings → Variables**: `ALLOWED_ORIGINS=https://<vercel-domain>`.

## 2. Frontend → Vercel

1. vercel.com → sign up **with GitHub** → Add New Project → import
   `movieslike` → Root Directory: `frontend/`.
2. Environment variables:
   - `NEXT_PUBLIC_API_URL` = `https://<HF_USER>-movieslike-api.hf.space`
   - `NEXT_PUBLIC_ENABLE_PROBE` = `false`
3. Deploy, note the domain, and feed it back into `ALLOWED_ORIGINS` (step 1.5).

## Gotchas

- Free Spaces sleep after ~48h idle; the wake-up takes ~1 min (plus model
  re-download on a fresh boot). Warm your link before interviews.
- To update the live API: re-run `pipeline/export_deploy_artifacts.py` if
  vectors changed, re-copy the files into the Space checkout, push.
- The Gradio demo UI and the JSON API share ranking code — `/docs` on the
  Space host shows the OpenAPI schema.
