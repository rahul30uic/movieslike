"use client";

import { useState } from "react";

/**
 * Free-text vibe search: the user describes the head-space in words and we
 * search the caption block of the hybrid embedding space.
 */
export const VibeSearch = ({ onSearch, isLoading }) => {
    const [query, setQuery] = useState("");

    const submit = (e) => {
        e.preventDefault();
        if (!isLoading) onSearch(query);
    };

    return (
        <form onSubmit={submit} className="flex gap-3">
            <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder='"cozy rainy night, gentle loneliness" · "characters who are batshit crazy"'
                className="flex-1 px-5 py-3.5 rounded-xl bg-stone-950/80 border border-stone-800 text-stone-200 placeholder-stone-600 focus:border-amber-400/70 focus:outline-none transition-colors duration-300"
            />
            <button
                type="submit"
                disabled={isLoading || query.trim().length < 2}
                className="px-6 py-3 bg-amber-400 text-stone-950 font-semibold rounded-xl hover:bg-amber-300 transition-all duration-300 disabled:bg-stone-800 disabled:text-stone-600 disabled:cursor-not-allowed whitespace-nowrap"
            >
                {isLoading ? "searching…" : "Find movies"}
            </button>
        </form>
    );
};
