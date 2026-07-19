"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

/**
 * The vibe-space atlas: a UMAP projection of the embedding space, rendered
 * with the actual mood images. Neighborhoods are real — foggy coasts cluster
 * together because the model placed them together.
 */
export default function AtlasPage() {
    const [points, setPoints] = useState(null);
    const [zoom, setZoom] = useState(1);
    const [hover, setHover] = useState(null);
    const scrollRef = useRef(null);

    useEffect(() => {
        fetch("/engine/atlas.json")
            .then((r) => r.json())
            .then((d) => setPoints(d.points))
            .catch(() => setPoints([]));
    }, []);

    const W = 2600, H = 1700;

    return (
        <main className="min-h-screen p-4 sm:p-8">
            <div className="max-w-6xl mx-auto">
                <header className="text-center mt-4 mb-6">
                    <h1 className="font-display text-4xl text-stone-100 tracking-tight">
                        <Link href="/">Movieslike</Link> <span className="text-stone-500">· the vibe atlas</span>
                    </h1>
                    <p className="text-stone-400 mt-3 max-w-2xl mx-auto text-sm">
                        Every mood in the corpus, placed by the model — a 2D projection (UMAP) of the
                        embedding space the recommendations live in. Nothing is hand-arranged:
                        neighborhoods exist because the model put similar feelings near each other.
                        Scroll, zoom, hover.
                    </p>
                    <div className="mt-4 flex items-center justify-center gap-2">
                        {[0.6, 1, 1.6, 2.4].map((z) => (
                            <button
                                key={z}
                                onClick={() => setZoom(z)}
                                className={`px-3 py-1 rounded-full text-xs border transition-colors ${
                                    zoom === z
                                        ? "border-amber-400 text-amber-200"
                                        : "border-stone-700 text-stone-400 hover:border-stone-500"
                                }`}
                            >
                                {z}×
                            </button>
                        ))}
                    </div>
                </header>

                {hover && (
                    <p className="text-center text-stone-400 font-display italic text-sm mb-3 min-h-5">
                        {hover}
                    </p>
                )}

                <div
                    ref={scrollRef}
                    className="rounded-2xl border border-stone-800 overflow-auto bg-stone-950/60"
                    style={{ height: "70vh" }}
                >
                    {!points && (
                        <p className="text-center text-stone-600 py-24 font-display italic">
                            charting the space…
                        </p>
                    )}
                    {points && (
                        <div
                            className="relative"
                            style={{ width: W * zoom, height: H * zoom }}
                        >
                            {points.map((p, i) => (
                                <img
                                    key={i}
                                    src={`/probe_imgs/${p.f}`}
                                    alt=""
                                    loading="lazy"
                                    onMouseEnter={() => setHover(p.c)}
                                    onMouseLeave={() => setHover(null)}
                                    className="absolute rounded-md border border-black/60 hover:border-amber-400 hover:z-20 hover:scale-[2.2] transition-transform duration-200 shadow-lg shadow-black/50 object-cover"
                                    style={{
                                        left: p.x * (W - 80) * zoom,
                                        top: p.y * (H - 80) * zoom,
                                        width: 64 * Math.max(zoom * 0.75, 0.7),
                                        height: 44 * Math.max(zoom * 0.75, 0.7),
                                    }}
                                />
                            ))}
                        </div>
                    )}
                </div>

                <p className="text-center text-stone-700 text-xs mt-6 mb-4">
                    500 posts, farthest-point sampled from 3,715 · vectors: [bge(vibe caption) ; SigLIP(image)] ·{" "}
                    <a
                        href="https://github.com/rahul30uic/movieslike"
                        className="hover:text-stone-500 underline underline-offset-2"
                    >
                        method &amp; code
                    </a>
                </p>
            </div>
        </main>
    );
}
