"use client";

import { useEffect, useState } from "react";
import { loadDetails } from "@/lib/details";

const RatingBadge = ({ label, value, color, href }) => {
    if (!value) return null;
    const inner = (
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-sm font-semibold ${color}`}>
            <span className="opacity-80 text-xs">{label}</span>
            <span>{value}</span>
        </div>
    );
    return href ? <a href={href} target="_blank" rel="noreferrer">{inner}</a> : inner;
};

export const MovieModal = ({ movie, onClose }) => {
    const [detail, setDetail] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!movie) return;
        setLoading(true);
        setDetail(null);
        loadDetails().then((map) => {
            setDetail(map[movie.tmdb_id] || {});
            setLoading(false);
        });
    }, [movie]);

    useEffect(() => {
        const onKey = (e) => e.key === "Escape" && onClose();
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [onClose]);

    if (!movie) return null;
    const d = detail || {};

    return (
        <div
            onClick={onClose}
            className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-4"
        >
            <div
                onClick={(e) => e.stopPropagation()}
                className="bg-stone-950 border border-stone-800 rounded-2xl w-full max-w-3xl max-h-[90vh] overflow-y-auto shadow-2xl shadow-black/60"
            >
                <div className="flex flex-col sm:flex-row gap-5 p-5">
                    {movie.poster_path ? (
                        <img
                            src={`https://image.tmdb.org/t/p/w342${movie.poster_path}`}
                            alt={movie.title}
                            className="w-40 shrink-0 rounded-lg self-start object-cover"
                        />
                    ) : null}
                    <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-3">
                            <h2 className="font-display text-2xl text-stone-100">{movie.title}</h2>
                            <button
                                onClick={onClose}
                                className="text-stone-500 hover:text-stone-200 text-2xl leading-none shrink-0"
                                aria-label="close"
                            >×</button>
                        </div>

                        <div className="flex flex-wrap gap-2 mt-3">
                            <RatingBadge label="IMDb" value={d.imdb_rating}
                                color="bg-amber-400/15 text-amber-300"
                                href={d.imdb_id ? `https://www.imdb.com/title/${d.imdb_id}` : null} />
                            <RatingBadge label="RT" value={d.rt_rating}
                                color="bg-red-500/15 text-red-300" />
                            <RatingBadge label="TMDB" value={d.tmdb_rating ? d.tmdb_rating.toFixed(1) : null}
                                color="bg-teal-500/15 text-teal-300" />
                        </div>

                        <p className="text-stone-400 text-sm leading-relaxed mt-4">
                            {loading ? "Loading…" : (d.overview || "No description available.")}
                        </p>
                    </div>
                </div>

                {d.trailer && (
                    <div className="px-5 pb-5">
                        <div className="relative w-full rounded-lg overflow-hidden" style={{ paddingTop: "56.25%" }}>
                            <iframe
                                className="absolute inset-0 w-full h-full"
                                src={`https://www.youtube.com/embed/${d.trailer}`}
                                title="trailer"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowFullScreen
                            />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};
