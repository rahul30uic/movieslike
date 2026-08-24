"use client";

// Lazily loads and caches the static movie-detail map (tmdb_id -> detail),
// fetched the first time a result card is opened.
let cache = null;
let inflight = null;

export async function loadDetails() {
    if (cache) return cache;
    if (!inflight) {
        inflight = fetch("/engine/movie_details.json")
            .then((r) => (r.ok ? r.json() : {}))
            .then((d) => { cache = d; return d; })
            .catch(() => ({}));
    }
    return inflight;
}
