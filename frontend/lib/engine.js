"use client";

/**
 * In-browser recommendation engine — no backend.
 *
 * Downloads quantized encoders (bge for text, SigLIP for images) via
 * transformers.js and a static movie index (fp16 vectors + metadata), then
 * does the same ranking as api/main.py: z-scored cosine, corpus-popularity
 * penalty, recognizability floor.
 *
 * The hybrid space is [768 caption-text ; 768 image] — text queries fill the
 * first block, image queries the second, zeros elsewhere.
 */

import { pipeline, AutoProcessor, SiglipVisionModel, RawImage } from "@huggingface/transformers";

const TEXT_MODEL = "Xenova/bge-base-en-v1.5";
const IMAGE_MODEL = "Xenova/siglip-base-patch16-224";
const QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";

let state = null;      // { movies, vectors, dim, penalties }
let textPipe = null;
let visionBits = null; // { processor, model }

// fp16 (uint16 bits) -> fp32
function halfToFloat(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

export async function loadIndex(onStatus) {
    if (state) return state;
    onStatus?.("Downloading movie index…");
    const [metaRes, binRes] = await Promise.all([
        fetch("/engine/movies.json"),
        fetch("/engine/vectors.bin"),
    ]);
    const meta = await metaRes.json();
    const buf = await binRes.arrayBuffer();
    const raw = new Uint16Array(buf);
    const vectors = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) vectors[i] = halfToFloat(raw[i]);

    // standardized log2(1 + n_posts), as in api/main.py
    const logPop = meta.movies.map((m) => Math.log2(1 + m.n));
    const mean = logPop.reduce((a, b) => a + b, 0) / logPop.length;
    const std = Math.sqrt(logPop.reduce((a, b) => a + (b - mean) ** 2, 0) / logPop.length) + 1e-9;
    const penalties = logPop.map((x) => (x - mean) / std);

    state = { movies: meta.movies, vectors, dim: meta.dim, penalties };
    return state;
}

function progressToStatus(onStatus, label) {
    return (p) => {
        if (p.status === "progress" && p.total) {
            onStatus?.(`Downloading ${label} (${Math.round((100 * p.loaded) / p.total)}%) — first visit only`);
        }
    };
}

export async function loadTextModel(onStatus) {
    if (!textPipe) {
        textPipe = await pipeline("feature-extraction", TEXT_MODEL, {
            dtype: "q8",
            progress_callback: progressToStatus(onStatus, "text model"),
        });
    }
    return textPipe;
}

export async function loadVisionModel(onStatus) {
    if (!visionBits) {
        const cb = progressToStatus(onStatus, "vision model");
        const [processor, model] = await Promise.all([
            AutoProcessor.from_pretrained(IMAGE_MODEL),
            SiglipVisionModel.from_pretrained(IMAGE_MODEL, { dtype: "q8", progress_callback: cb }),
        ]);
        visionBits = { processor, model };
    }
    return visionBits;
}

function normalize(arr) {
    let n = 0;
    for (const x of arr) n += x * x;
    n = Math.sqrt(n) + 1e-9;
    return Float32Array.from(arr, (x) => x / n);
}

// =============================================================================
// Taste vector: few-shot personalization. Completed sessions (probe
// posteriors, search targets) fold into a persistent per-device vector via
// EWMA; rankings blend tonight's intent with the accumulated taste prior.
// =============================================================================

const TASTE_KEY = "movieslike_taste_v1";
const TASTE_MOMENTUM = 0.85; // EWMA decay per update
const TASTE_BLEND = 0.2;     // weight of the prior in every ranking
const TASTE_MIN_UPDATES = 3; // don't personalize off nearly nothing

function loadTaste() {
    try {
        const raw = localStorage.getItem(TASTE_KEY);
        if (!raw) return null;
        const t = JSON.parse(raw);
        return { vec: Float32Array.from(t.vec), n: t.n };
    } catch {
        return null;
    }
}

function updateTaste(vec) {
    try {
        const cur = loadTaste();
        let merged;
        if (cur && cur.vec.length === vec.length) {
            merged = new Float32Array(vec.length);
            for (let j = 0; j < vec.length; j++) {
                merged[j] = TASTE_MOMENTUM * cur.vec[j] + (1 - TASTE_MOMENTUM) * vec[j];
            }
        } else {
            merged = Float32Array.from(vec);
        }
        merged = normalize(merged);
        localStorage.setItem(TASTE_KEY, JSON.stringify({
            vec: Array.from(merged, (x) => Math.round(x * 1e5) / 1e5),
            n: (cur?.n || 0) + 1,
        }));
    } catch {
        /* private browsing etc. — personalization is best-effort */
    }
}

function personalize(target) {
    const t = loadTaste();
    if (!t || t.n < TASTE_MIN_UPDATES) return target;
    const out = new Float32Array(target.length);
    for (let j = 0; j < target.length; j++) {
        out[j] = (1 - TASTE_BLEND) * target[j] + TASTE_BLEND * t.vec[j];
    }
    return normalize(out);
}

export function tasteInfo() {
    const t = loadTaste();
    return { active: !!t && t.n >= TASTE_MIN_UPDATES, sessions: t?.n || 0 };
}

export function resetTaste() {
    try {
        localStorage.removeItem(TASTE_KEY);
    } catch { /* ignore */ }
}

/** Ranking — mirror of rank() in api/main.py. */
export function rank(target, { alpha = 0.3, minVotes = 500, minSupport = 3, n = 12 } = {}) {
    const { movies, vectors, dim, penalties } = state;
    const sims = new Float32Array(movies.length);
    for (let i = 0; i < movies.length; i++) {
        let s = 0;
        const off = i * dim;
        for (let j = 0; j < dim; j++) s += vectors[off + j] * target[j];
        sims[i] = s;
    }
    const mean = sims.reduce((a, b) => a + b, 0) / sims.length;
    const std = Math.sqrt(sims.reduce((a, b) => a + (b - mean) ** 2, 0) / sims.length) + 1e-9;

    const scored = [];
    for (let i = 0; i < movies.length; i++) {
        const m = movies[i];
        if (m.n < minSupport || m.v < minVotes) continue;
        scored.push([(sims[i] - mean) / std - alpha * penalties[i], i]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    return scored.slice(0, n).map(([score, i]) => ({
        tmdb_id: state.movies[i].id,
        title: state.movies[i].t,
        poster_path: state.movies[i].p,
        n_posts: state.movies[i].n,
        score: Math.round(score * 1e4) / 1e4,
    }));
}

// =============================================================================
// Head-space probe (browser port of api/main.py's Bayesian probe):
// particles are well-supported movies; probe images are their TMDB backdrops.
// =============================================================================

const PROBE_SHARPNESS = 25.0;
const PROBE_ROUNDS = 5;
const PROBE_CANDIDATES = 24; // B-candidates scored per round (each costs a pool pass)

// Curated corpus posts (vibe-space farthest-point sample): the particles of
// the posterior AND the images shown. Vectors ship as a small fp16 bin.
let probePool = null; // { items: [{id, f}], vecs: Float32Array, dim }

async function loadProbe(onStatus) {
    if (!probePool) {
        await loadIndex(onStatus);
        onStatus?.("Loading probe images…");
        const [metaRes, binRes] = await Promise.all([
            fetch("/engine/probe_posts.json"),
            fetch("/engine/probe_posts.bin"),
        ]);
        const meta = await metaRes.json();
        const raw = new Uint16Array(await binRes.arrayBuffer());
        const vecs = new Float32Array(raw.length);
        for (let i = 0; i < raw.length; i++) vecs[i] = halfToFloat(raw[i]);
        probePool = { items: meta.posts, vecs, dim: meta.dim };
        onStatus?.(null);
    }
    return probePool;
}

function poolVec(p) {
    return probePool.vecs.subarray(p * probePool.dim, (p + 1) * probePool.dim);
}

function dot(a, b) {
    let s = 0;
    for (let j = 0; j < a.length; j++) s += a[j] * b[j];
    return s;
}

/** Bradley-Terry particle posterior over the probe pool. History references
 *  pool row indices (chosen/rejected).
 *
 *  Margins are z-scored across particles before the sigmoid: raw cosine
 *  differences in this space are tiny, so unstandardized likelihoods barely
 *  move the posterior and rounds FEEL independent. Standardized, one pick
 *  shifts a 1-sigma-preferring particle by ~4:1 odds. */
const PICK_DECISIVENESS = 1.4; // sigmoid slope per sigma of preference

function probePosterior(history) {
    const n = probePool.items.length;
    const w = new Float64Array(n).fill(1);
    const margins = new Float64Array(n);
    for (const { chosen, rejected } of history) {
        const vc = poolVec(chosen), vr = poolVec(rejected);
        let mean = 0;
        for (let p = 0; p < n; p++) {
            const v = poolVec(p);
            margins[p] = dot(v, vc) - dot(v, vr);
            mean += margins[p];
        }
        mean /= n;
        let varSum = 0;
        for (let p = 0; p < n; p++) varSum += (margins[p] - mean) ** 2;
        const std = Math.sqrt(varSum / n) + 1e-9;
        for (let p = 0; p < n; p++) {
            w[p] *= 1 / (1 + Math.exp(-PICK_DECISIVENESS * (margins[p] / std)));
        }
    }
    let total = 0;
    for (const x of w) total += x;
    for (let p = 0; p < n; p++) w[p] = total > 0 ? w[p] / total : 1 / n;
    return w;
}

/** Diagnosis-style summary: posterior confidence (1 - normalized entropy)
 *  and the current best "reading" of the mood. */
export async function probeSummary(history, onStatus) {
    await loadProbe(onStatus);
    const w = probePosterior(history);
    let H = 0;
    for (const x of w) if (x > 0) H -= x * Math.log(x);
    const confidence = 1 - H / Math.log(w.length);
    let top = 0;
    for (let p = 1; p < w.length; p++) if (w[p] > w[top]) top = p;
    return {
        confidence,
        reading: probePool.items[top].c || null,
    };
}

function weightedPick(indices, w) {
    let total = 0;
    for (const p of indices) total += w[p];
    let r = Math.random() * total;
    for (const p of indices) {
        r -= w[p];
        if (r <= 0) return p;
    }
    return indices[indices.length - 1];
}

export async function probeNext(history, onStatus) {
    await loadProbe(onStatus);
    const n = probePool.items.length;
    const w = probePosterior(history);
    const shown = new Set(history.flatMap((h) => [h.chosen, h.rejected]));
    const avail = [];
    for (let p = 0; p < n; p++) if (!shown.has(p)) avail.push(p);

    // A: sampled from the posterior. B: among posterior-weighted candidates,
    // far from A and predicted ~50/50 (cheap expected-information-gain proxy).
    const a = weightedPick(avail, w);
    const va = poolVec(a);
    const simsA = new Float32Array(n);
    for (let p = 0; p < n; p++) simsA[p] = dot(poolVec(p), va);

    const cands = new Set();
    for (let k = 0; k < PROBE_CANDIDATES * 3 && cands.size < PROBE_CANDIDATES; k++) {
        const c = weightedPick(avail, w);
        if (c !== a) cands.add(c);
    }
    // Coarse-to-fine schedule: early rounds demand maximum contrast (broad
    // differential); later rounds relax it so both images come from inside
    // the posterior's region — fine-grained follow-up questions.
    const progress = Math.min(history.length / (PROBE_ROUNDS - 1), 1);
    const contrastWeight = 0.8 * (1 - progress) + 0.1;

    let best = null, bestScore = Infinity;
    for (const c of cands) {
        const vc = poolVec(c);
        const simAB = dot(va, vc);
        let pA = 0;
        for (let p = 0; p < n; p++) {
            const margin = simsA[p] - dot(poolVec(p), vc);
            pA += w[p] / (1 + Math.exp(-PROBE_SHARPNESS * margin));
        }
        const score = Math.abs(pA - 0.5) + contrastWeight * simAB;
        if (score < bestScore) { bestScore = score; best = c; }
    }

    const toImg = (p) => ({
        post_id: p, // pool row index is the id the UI hands back in history
        image_url: "/probe_imgs/" + probePool.items[p].f,
    });
    return { round: history.length + 1, total_rounds: PROBE_ROUNDS, pair: [toImg(a), toImg(best)] };
}

export async function probeRecommend(history, opts, onStatus) {
    await loadProbe(onStatus);
    const w = probePosterior(history);
    const target = new Float32Array(state.dim);
    for (let p = 0; p < probePool.items.length; p++) {
        const v = poolVec(p);
        for (let j = 0; j < state.dim; j++) target[j] += w[p] * v[j];
    }
    updateTaste(normalize(target)); // a completed diagnosis is the strongest taste signal
    return rank(personalize(target), opts);
}

export async function searchVector(vec, opts, onStatus) {
    await loadIndex(onStatus);
    const target = new Float32Array(state.dim);
    target.set(vec.slice(0, state.dim), 0);
    return rank(target, opts);
}

export async function searchText(query, opts, onStatus) {
    await loadIndex(onStatus);
    const pipe = await loadTextModel(onStatus);
    onStatus?.("Embedding your words…");
    const out = await pipe(QUERY_PREFIX + query, { pooling: "cls", normalize: true });
    const q = out.data; // 768, unit norm
    const target = new Float32Array(state.dim);
    target.set(q, 0); // caption block; image block stays zero
    onStatus?.(null);
    updateTaste(target);
    return rank(personalize(target), opts);
}

export async function searchImage(fileOrUrl, opts, onStatus) {
    await loadIndex(onStatus);
    const { processor, model } = await loadVisionModel(onStatus);
    onStatus?.("Embedding your image…");
    const url = typeof fileOrUrl === "string" ? fileOrUrl : URL.createObjectURL(fileOrUrl);
    const image = await RawImage.read(url);
    const inputs = await processor(image);
    const out = await model(inputs);
    const v = normalize(out.pooler_output.data); // = SigLIP get_image_features
    const target = new Float32Array(state.dim);
    target.set(v, state.dim - v.length); // image block
    onStatus?.(null);
    updateTaste(target);
    const recommendations = rank(personalize(target), opts);

    // --- Introspection: which regions of the upload drove the match? ---
    // SigLIP patch tokens vs the top match's image block, as a spatial grid.
    let explanation = null;
    const patches = out.last_hidden_state; // [1, n_patches, 768]
    if (patches && recommendations.length > 0) {
        const [, nPatch, pDim] = patches.dims;
        const half = state.dim / 2;
        // First recommendation whose image block actually carries signal
        let refVec = null, refTitle = null;
        for (const rec of recommendations) {
            const i = state.movies.findIndex((m) => m.id === rec.tmdb_id);
            if (i < 0) continue;
            const block = state.vectors.subarray(i * state.dim + half, (i + 1) * state.dim);
            let norm = 0;
            for (const x of block) norm += x * x;
            if (Math.sqrt(norm) > 0.1) {
                refVec = normalize(block);
                refTitle = rec.title;
                break;
            }
        }
        if (refVec) {
            const heat = new Float32Array(nPatch);
            for (let j = 0; j < nPatch; j++) {
                const patch = normalize(patches.data.subarray(j * pDim, (j + 1) * pDim));
                heat[j] = dot(patch, refVec);
            }
            let lo = Infinity, hi = -Infinity;
            for (const x of heat) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
            const range = hi - lo || 1;
            const grid = Array.from(heat, (x) => ((x - lo) / range) ** 2); // gamma-sharpened
            const side = Math.round(Math.sqrt(nPatch));
            explanation = { grid, cols: side, rows: side, vs: refTitle, imageUrl: url };
        }
    }

    return { recommendations, explanation };
}
