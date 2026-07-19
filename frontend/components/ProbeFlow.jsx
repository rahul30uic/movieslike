"use client";

import { useCallback, useEffect, useState } from "react";
import { MovieGrid } from "@/components/MovieGrid";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const BROWSER_ENGINE = process.env.NEXT_PUBLIC_USE_BROWSER_ENGINE === "true";

/**
 * The head-space probe: five rounds of "which feels more like tonight?",
 * then exactly five movies. The signature interaction — no words required.
 */
export const ProbeFlow = () => {
    const [history, setHistory] = useState([]);
    const [pair, setPair] = useState(null);
    const [round, setRound] = useState(1);
    const [totalRounds, setTotalRounds] = useState(5);
    const [movies, setMovies] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    // Diagnosis readout: posterior confidence + current best mood reading
    const [summary, setSummary] = useState(null);
    const [taste, setTaste] = useState(null);

    const refreshSummary = useCallback(async (hist) => {
        if (!BROWSER_ENGINE || hist.length === 0) return;
        try {
            const engine = await import("@/lib/engine");
            setSummary(await engine.probeSummary(hist, () => {}));
        } catch {
            /* readout is cosmetic — never block the flow */
        }
    }, []);

    const fetchNextPair = useCallback(async (hist) => {
        setIsLoading(true);
        setError(null);
        try {
            let data;
            if (BROWSER_ENGINE) {
                const engine = await import("@/lib/engine");
                data = await engine.probeNext(hist, () => {});
            } else {
                const res = await fetch(`${API_BASE_URL}/probe/next`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ history: hist }),
                });
                if (!res.ok) throw new Error(`Probe request failed: ${res.statusText}`);
                data = await res.json();
            }
            setPair(data.pair);
            setRound(data.round);
            setTotalRounds(data.total_rounds);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    }, []);

    const fetchRecommendations = useCallback(async (hist) => {
        setIsLoading(true);
        setError(null);
        try {
            let recs;
            if (BROWSER_ENGINE) {
                const engine = await import("@/lib/engine");
                recs = await engine.probeRecommend(hist, { alpha: 0.3, minVotes: 500, n: 5 }, () => {});
            } else {
                const res = await fetch(`${API_BASE_URL}/probe/recommend`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ history: hist, alpha: 0.3, num_recommendations: 5, min_votes: 500 }),
                });
                if (!res.ok) throw new Error(`Recommendation failed: ${res.statusText}`);
                recs = (await res.json()).recommendations || [];
            }
            setMovies(recs);
            if (BROWSER_ENGINE) {
                const engine = await import("@/lib/engine");
                setTaste(engine.tasteInfo());
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchNextPair([]);
    }, [fetchNextPair]);

    const pick = (chosen, rejected) => {
        if (isLoading) return;
        const hist = [...history, { chosen: chosen.post_id, rejected: rejected.post_id }];
        setHistory(hist);
        refreshSummary(hist);
        if (hist.length >= totalRounds) {
            setPair(null);
            fetchRecommendations(hist);
        } else {
            fetchNextPair(hist);
        }
    };

    const restart = () => {
        setHistory([]);
        setMovies(null);
        setPair(null);
        setSummary(null);
        fetchNextPair([]);
    };

    return (
        <div>
            {error && (
                <div className="text-center text-red-400/90 mb-8 text-sm">{error}</div>
            )}

            {!movies && (
                <>
                    <div className="max-w-md mx-auto mb-3">
                        <div className="flex justify-between items-baseline mb-1.5">
                            <span className="text-stone-500 text-xs tracking-widest uppercase">mood lock</span>
                            <span className="text-stone-500 text-xs tracking-widest uppercase">
                                {Math.min(round, totalRounds)} / {totalRounds}
                            </span>
                        </div>
                        <div className="h-1.5 rounded-full bg-stone-800 overflow-hidden">
                            <div
                                className="h-full bg-amber-400 rounded-full transition-all duration-700"
                                style={{ width: `${Math.round(100 * (summary?.confidence ?? 0))}%` }}
                            />
                        </div>
                    </div>
                    <p className="text-center text-stone-500 font-display italic text-sm mb-7 min-h-5 transition-opacity duration-500">
                        {summary?.reading ? `current reading: ${summary.reading}` : " "}
                    </p>

                    {pair && !isLoading ? (
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                            {pair.map((img, idx) => (
                                <button
                                    key={img.post_id}
                                    onClick={() => pick(img, pair[1 - idx])}
                                    className="group relative rounded-2xl overflow-hidden border border-stone-800 hover:border-amber-400/70 focus:border-amber-400 transition-all duration-300 hover:scale-[1.01] cursor-pointer shadow-2xl shadow-black/60"
                                >
                                    <img
                                        src={img.image_url.startsWith("http") ? img.image_url : encodeURI(img.image_url)}
                                        alt="a mood"
                                        className="w-full h-72 sm:h-96 object-cover saturate-[0.92] group-hover:saturate-110 transition-all duration-500"
                                    />
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent opacity-60 group-hover:opacity-90 transition-opacity duration-300 flex items-end p-5">
                                        <span className="text-amber-200/0 group-hover:text-amber-200 font-display italic text-lg transition-colors duration-300">
                                            this one
                                        </span>
                                    </div>
                                </button>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center text-stone-600 py-24 font-display italic text-lg">
                            {isLoading ? "reading the room…" : null}
                        </div>
                    )}
                </>
            )}

            {movies && (
                <>
                    <h2 className="font-display text-2xl sm:text-3xl text-center text-stone-200 mb-3">
                        Your five for tonight.
                    </h2>
                    {summary?.reading && (
                        <p className="text-center text-stone-500 font-display italic text-sm mb-8">
                            diagnosis: {summary.reading}
                        </p>
                    )}
                    <MovieGrid movies={movies} isLoading={isLoading} error={null} />
                    <div className="text-center mt-10">
                        <button
                            onClick={restart}
                            className="px-6 py-2.5 border border-stone-700 text-stone-300 rounded-full hover:border-amber-400/60 hover:text-amber-200 transition-colors duration-300 text-sm"
                        >
                            different mood, start over
                        </button>
                        {taste?.active && (
                            <p className="text-stone-600 text-xs mt-4">
                                lightly personalized by {taste.sessions} past sessions on this device ·{" "}
                                <button
                                    onClick={async () => {
                                        const engine = await import("@/lib/engine");
                                        engine.resetTaste();
                                        setTaste(engine.tasteInfo());
                                    }}
                                    className="underline underline-offset-2 hover:text-stone-400"
                                >
                                    forget my taste
                                </button>
                            </p>
                        )}
                    </div>
                </>
            )}
        </div>
    );
};
