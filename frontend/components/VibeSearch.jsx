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
                placeholder='Describe the vibe… e.g. "cozy rainy night, gentle loneliness" or "batshit crazy characters"'
                className="flex-1 px-4 py-3 rounded-lg bg-gray-800 border-2 border-gray-700 text-white placeholder-gray-500 focus:border-purple-500 focus:outline-none transition-colors duration-300"
            />
            <button
                type="submit"
                disabled={isLoading || query.trim().length < 2}
                className="px-6 py-3 bg-purple-600 text-white font-bold rounded-lg shadow-lg hover:bg-purple-700 transition-all duration-300 disabled:bg-gray-600 disabled:cursor-not-allowed disabled:shadow-none whitespace-nowrap"
            >
                {isLoading ? "Searching…" : "Search Vibe"}
            </button>
        </form>
    );
};
