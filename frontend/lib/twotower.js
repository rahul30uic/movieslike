"use client";

/**
 * Two-stage serving: two-tower retrieval + MMR reranking, all in-browser.
 *
 * Stage 1 (retrieval): a query's 1536-d hybrid vector -> query tower (a small
 * MLP, weights shipped as fp16) -> 256-d -> cosine over the 256-d item index
 * -> top-K candidates.
 * Stage 2 (rerank): popularity-debiased relevance + MMR diversity + a votes
 * floor -> the final N. This is where the long-tail/diversity policy lives,
 * separate from retrieval (see eval/reranker_eval.py).
 */

let tt = null; // { items, dim, movies, w1,b1,w2,b2, hDim, inDim, outDim }

function halfToFloat(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function f16buf(ab) {
    const raw = new Uint16Array(ab);
    const out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) out[i] = halfToFloat(raw[i]);
    return out;
}

export async function loadTwoTower(onStatus) {
    if (tt) return tt;
    onStatus?.("Loading recommender…");
    const [mRes, iRes, wRes, sRes] = await Promise.all([
        fetch("/engine/tt_movies.json"),
        fetch("/engine/tt_items.bin"),
        fetch("/engine/tt_query_tower.bin"),
        fetch("/engine/tt_query_tower.json"),
    ]);
    const meta = await mRes.json();
    const items = f16buf(await iRes.arrayBuffer());
    const wf = f16buf(await wRes.arrayBuffer());
    const shapes = (await sRes.json()).shapes;

    let o = 0;
    const take = (n) => { const s = wf.subarray(o, o + n); o += n; return s; };
    const w1 = take(shapes.w1[0] * shapes.w1[1]);
    const b1 = take(shapes.b1[0]);
    const w2 = take(shapes.w2[0] * shapes.w2[1]);
    const b2 = take(shapes.b2[0]);

    tt = { items, dim: meta.dim, movies: meta.movies, w1, b1, w2, b2,
           hDim: shapes.w1[0], inDim: shapes.w1[1], outDim: shapes.w2[0] };
    onStatus?.(null);
    return tt;
}

function gelu(x) {
    return 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

/** Query tower forward: 1536-d hybrid -> 256-d, L2-normalized. */
export function queryTower(x) {
    const { w1, b1, w2, b2, hDim, inDim, outDim } = tt;
    const h = new Float32Array(hDim);
    for (let i = 0; i < hDim; i++) {
        let s = b1[i];
        const off = i * inDim;
        for (let j = 0; j < inDim; j++) s += w1[off + j] * x[j];
        h[i] = gelu(s);
    }
    const out = new Float32Array(outDim);
    let nrm = 0;
    for (let i = 0; i < outDim; i++) {
        let s = b2[i];
        const off = i * hDim;
        for (let j = 0; j < hDim; j++) s += w2[off + j] * h[j];
        out[i] = s;
        nrm += s * s;
    }
    nrm = Math.sqrt(nrm) + 1e-9;
    for (let i = 0; i < outDim; i++) out[i] /= nrm;
    return out;
}

function zscore(arr) {
    const m = arr.reduce((a, b) => a + b, 0) / arr.length;
    const sd = Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length) + 1e-9;
    return arr.map((x) => (x - m) / sd);
}

/**
 * Two-stage recommend. `target` is the 1536-d hybrid query vector (same as the
 * old rank() input), so all query paths (text/image/probe) work unchanged.
 * alpha = popularity de-bias (hidden gems ↔ favorites), lambda = MMR diversity.
 */
export function rankTwoTower(target, { alpha = 0.3, lambda = 0.3, minVotes = 500, n = 12, K = 200 } = {}) {
    const { items, dim, movies } = tt;
    const q = queryTower(target);
    const M = movies.length;

    const sims = new Float32Array(M);
    for (let i = 0; i < M; i++) {
        let s = 0;
        const off = i * dim;
        for (let j = 0; j < dim; j++) s += items[off + j] * q[j];
        sims[i] = s;
    }

    let idx = [];
    for (let i = 0; i < M; i++) if (movies[i].v >= minVotes) idx.push(i);
    idx.sort((a, b) => sims[b] - sims[a]);
    const cand = idx.slice(0, K);

    const zr = zscore(cand.map((i) => sims[i]));
    const zp = zscore(cand.map((i) => Math.log1p(movies[i].n)));
    let base = cand.map((_, k) => zr[k] - alpha * zp[k]);
    const mn = Math.min(...base), mx = Math.max(...base);
    base = base.map((b) => (b - mn) / (mx - mn + 1e-9));

    const candVec = (k) => items.subarray(cand[k] * dim, (cand[k] + 1) * dim);
    const selected = [];
    const remaining = cand.map((_, k) => k);
    while (selected.length < n && remaining.length) {
        let best = -1, bestVal = -Infinity;
        for (const r of remaining) {
            let div = 0;
            const vr = candVec(r);
            for (const sIdx of selected) {
                const vs = candVec(sIdx);
                let d = 0;
                for (let j = 0; j < dim; j++) d += vr[j] * vs[j];
                if (d > div) div = d;
            }
            const val = (1 - lambda) * base[r] - lambda * div;
            if (val > bestVal) { bestVal = val; best = r; }
        }
        selected.push(best);
        remaining.splice(remaining.indexOf(best), 1);
    }

    return selected.map((k) => {
        const m = movies[cand[k]];
        return { tmdb_id: m.id, title: m.t, poster_path: m.p, n_posts: m.n,
                 score: Math.round(base[k] * 1e4) / 1e4 };
    });
}
