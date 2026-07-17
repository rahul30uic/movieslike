# Deploying the live demo

**Current architecture: no backend.** All inference runs in the visitor's
browser — transformers.js downloads quantized encoders (bge for text, SigLIP
for images), the movie index ships as static files
(`frontend/public/engine/`), and ranking is plain JS
(`frontend/lib/engine.js`). Vercel free tier hosts everything. $0, no cold
starts, no CORS, nothing to keep alive.

(History: v1 targeted a HF Docker Space — went paid in 2026. v2 targeted a
Gradio Space — also paid now. `deploy/hf_space/` + `deploy/Dockerfile` remain
for anyone with HF PRO; the FastAPI in `api/` is still the local dev server
and the only way to use the head-space probe.)

## Vercel setup (one time)

1. vercel.com → sign up **with GitHub** → Add New → Project → import
   `movieslike`.
2. Root Directory: `frontend/`.
3. Environment variables:
   - `NEXT_PUBLIC_USE_BROWSER_ENGINE` = `true`
   - `NEXT_PUBLIC_ENABLE_PROBE` = `false`
4. Deploy. The assigned domain is the resume link.

Pushes to the repo's default branch redeploy automatically.

## Updating the index

After pipeline changes: `python pipeline/export_web_index.py`, commit the
regenerated `frontend/public/engine/`, push.

## Notes

- First search downloads ~140MB of quantized models (progress shown in the
  UI); the browser caches them afterwards. The 29MB index loads per session.
- Embedding parity JS-vs-Python was verified (`frontend/scripts/parity_test.mjs`):
  text cosine 0.99; image 0.94 due to resize-interpolation differences,
  which still yields 6/8 identical top-8 retrievals.
- Probe stays local-only: it needs the 3.5GB Reddit image corpus, which is
  neither shipped nor redistributable.
