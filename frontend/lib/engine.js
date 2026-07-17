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
    return rank(target, opts);
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
    return rank(target, opts);
}
