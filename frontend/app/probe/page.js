"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { MovieGrid } from "@/components/MovieGrid";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Head-space probe: five rounds of "which of these two feels more like
 * tonight?" Each pick narrows a posterior over vibe space; at the end the
 * user gets exactly five movies.
 */
export default function ProbePage() {
    const [history, setHistory] = useState([]);
    const [pair, setPair] = useState(null);
    const [round, setRound] = useState(1);
    const [totalRounds, setTotalRounds] = useState(5);
    const [movies, setMovies] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchNextPair = useCallback(async (hist) => {
        setIsLoading(true);
        setError(null);
        try {
            const res = await fetch(`${API_BASE_URL}/probe/next`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ history: hist }),
            });
            if (!res.ok) throw new Error(`Probe request failed: ${res.statusText}`);
            const data = await res.json();
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
            const res = await fetch(`${API_BASE_URL}/probe/recommend`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                // min_votes: only recommend movies enough people have seen to
                // rate on TMDB — the probe should end on findable titles.
                body: JSON.stringify({ history: hist, alpha: 0.3, num_recommendations: 5, min_votes: 500 }),
            });
            if (!res.ok) throw new Error(`Recommendation failed: ${res.statusText}`);
            const data = await res.json();
            setMovies(data.recommendations || []);
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
        fetchNextPair([]);
    };

    return (
        <main className="min-h-screen bg-gray-900 text-white p-4 sm:p-8">
            <div className="max-w-5xl mx-auto">
                <header className="text-center mb-10">
                    <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500">
                        Lock In Your Head-Space
                    </h1>
                    <p className="text-gray-400 mt-2">
                        {movies
                            ? "Your five movies. No scrolling required."
                            : "Which of these feels more like tonight? No words needed — just react."}
                    </p>
                    <Link href="/" className="text-sm text-purple-400 hover:text-purple-300 mt-2 inline-block">
                        ← back to search
                    </Link>
                </header>

                {error && (
                    <div className="text-center text-red-400 mb-8">
                        {error} — is the backend running on :8000?
                    </div>
                )}

                {!movies && (
                    <>
                        <div className="flex items-center justify-center gap-2 mb-8">
                            {Array.from({ length: totalRounds }).map((_, i) => (
                                <div
                                    key={i}
                                    className={`h-2 w-10 rounded-full transition-colors duration-300 ${
                                        i < history.length ? "bg-purple-500" : "bg-gray-700"
                                    }`}
                                />
                            ))}
                            <span className="ml-3 text-gray-400 text-sm">
                                {Math.min(round, totalRounds)} / {totalRounds}
                            </span>
                        </div>

                        {pair && !isLoading ? (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                                {pair.map((img, idx) => (
                                    <button
                                        key={img.post_id}
                                        onClick={() => pick(img, pair[1 - idx])}
                                        className="group relative rounded-xl overflow-hidden border-2 border-gray-700 hover:border-purple-500 focus:border-purple-500 transition-all duration-300 hover:scale-[1.01] cursor-pointer"
                                    >
                                        <img
                                            src={encodeURI(img.image_url)}
                                            alt="mood option"
                                            className="w-full h-96 object-cover"
                                        />
                                        <div className="absolute inset-0 bg-gradient-to-t from-gray-900/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end p-4">
                                            <span className="text-white font-semibold">This one</span>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        ) : (
                            <div className="text-center text-gray-500 py-24">
                                {isLoading ? "Reading the room…" : null}
                            </div>
                        )}
                    </>
                )}

                {movies && (
                    <>
                        <MovieGrid movies={movies} isLoading={isLoading} error={null} />
                        <div className="text-center mt-10">
                            <button
                                onClick={restart}
                                className="px-6 py-3 bg-gray-700 text-white font-semibold rounded-lg hover:bg-gray-600 transition-colors duration-300"
                            >
                                Start over
                            </button>
                        </div>
                    </>
                )}
            </div>
        </main>
    );
}
