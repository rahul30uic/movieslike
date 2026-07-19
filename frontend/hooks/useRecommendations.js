"use client";

import { useState, useCallback } from 'react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
// When true, all inference runs in the browser (transformers.js + static
// index) — no backend at all. Set on Vercel; local dev uses the FastAPI.
const BROWSER_ENGINE = process.env.NEXT_PUBLIC_USE_BROWSER_ENGINE === "true";

/**
 * Custom hook to manage recommendation state and logic.
 * This hook encapsulates all interactions with the backend API.
 *
 * @returns {object} The state and functions to interact with the recommendations.
 */
export const useRecommendations = () => {
    // State for the currently selected anchor vector
    const [selectedAnchor, setSelectedAnchor] = useState(null);
    
    // State for the popularity penalty factor
    const [alpha, setAlpha] = useState(0.5);
    
    // State for the API response
    const [movies, setMovies] = useState([]);
    
    // State for loading and error handling
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    // Progress line for the in-browser engine (model downloads etc.)
    const [engineStatus, setEngineStatus] = useState(null);

    // Patch-level heatmap explaining an image query's top match
    const [explanation, setExplanation] = useState(null);

    // Last executed search, so "Regenerate" can re-run it after the user
    // moves the slider (or otherwise changes settings) without re-entering
    // the query or re-uploading the image.
    const [lastSearch, setLastSearch] = useState(null); // {type: 'text'|'image', payload}

    /** Runs an in-browser engine call with shared load/error handling. */
    const runEngine = useCallback(async (fn) => {
        setIsLoading(true);
        setError(null);
        setMovies([]);
        try {
            const engine = await import("@/lib/engine");
            const recs = await fn(engine);
            setMovies(recs);
        } catch (err) {
            setError(err.message || "In-browser engine failed.");
        } finally {
            setEngineStatus(null);
            setIsLoading(false);
        }
    }, []);

    /**
     * Fetches movie recommendations from the FastAPI backend.
     * This function is explicitly triggered by the UI.
     */
    const fetchRecommendations = useCallback(async () => {
        if (!selectedAnchor) {
            setError("Please select a vibe anchor first.");
            return;
        }

        if (BROWSER_ENGINE) {
            return runEngine((e) =>
                e.searchVector(selectedAnchor, { alpha, minVotes: 500, n: 12 }, setEngineStatus));
        }

        setIsLoading(true);
        setError(null);
        setMovies([]); // Clear previous results

        try {
            const response = await fetch(`${API_BASE_URL}/recommendations`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    target_vector: selectedAnchor,
                    alpha: alpha,
                    top_k: 30, // The initial candidate pool size
                    num_recommendations: 12 // How many final results we want
                }),
            });

            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.statusText}`);
            }

            const data = await response.json();
            setMovies(data.recommendations || []);
        } catch (err) {
            setError(err.message || "An unknown error occurred while fetching recommendations.");
        } finally {
            setIsLoading(false);
        }
    }, [selectedAnchor, alpha]); // Dependencies for the fetch function

    /**
     * Fetches recommendations from a free-text vibe description.
     * Clears any selected anchor — the query becomes the target.
     */
    const fetchByText = useCallback(async (query) => {
        if (!query || query.trim().length < 2) {
            setError("Describe the vibe in a few words first.");
            return;
        }
        setLastSearch({ type: "text", payload: query.trim() });

        if (BROWSER_ENGINE) {
            setSelectedAnchor(null);
            setExplanation(null);
            return runEngine((e) =>
                e.searchText(query.trim(), { alpha, minVotes: 500, n: 12 }, setEngineStatus));
        }

        setSelectedAnchor(null);
        setIsLoading(true);
        setError(null);
        setMovies([]);

        try {
            const response = await fetch(`${API_BASE_URL}/recommendations/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query.trim(),
                    alpha: alpha,
                    num_recommendations: 12
                }),
            });

            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.statusText}`);
            }

            const data = await response.json();
            setMovies(data.recommendations || []);
        } catch (err) {
            setError(err.message || "An unknown error occurred while fetching recommendations.");
        } finally {
            setIsLoading(false);
        }
    }, [alpha]);

    /**
     * Fetches recommendations from an uploaded mood image (query-by-image).
     */
    const fetchByImage = useCallback(async (file) => {
        if (!file) {
            setError("Choose an image first.");
            return;
        }
        setLastSearch({ type: "image", payload: file });

        if (BROWSER_ENGINE) {
            setSelectedAnchor(null);
            setExplanation(null);
            return runEngine(async (e) => {
                const res = await e.searchImage(file, { alpha, minVotes: 500, n: 12 }, setEngineStatus);
                setExplanation(res.explanation);
                return res.recommendations;
            });
        }

        setSelectedAnchor(null);
        setIsLoading(true);
        setError(null);
        setMovies([]);

        try {
            const form = new FormData();
            form.append("file", file);
            form.append("alpha", String(alpha));
            form.append("num_recommendations", "12");

            const response = await fetch(`${API_BASE_URL}/recommendations/image`, {
                method: 'POST',
                body: form,
            });

            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.statusText}`);
            }

            const data = await response.json();
            setMovies(data.recommendations || []);
        } catch (err) {
            setError(err.message || "An unknown error occurred while fetching recommendations.");
        } finally {
            setIsLoading(false);
        }
    }, [alpha]);

    /** Re-runs the last search with the current settings (alpha etc.). */
    const regenerate = useCallback(() => {
        if (!lastSearch || isLoading) return;
        if (lastSearch.type === "text") return fetchByText(lastSearch.payload);
        return fetchByImage(lastSearch.payload);
    }, [lastSearch, isLoading, fetchByText, fetchByImage]);

    return {
        selectedAnchor, setSelectedAnchor, alpha, setAlpha, movies, isLoading,
        error, engineStatus, explanation, hasLastSearch: !!lastSearch,
        fetchRecommendations, fetchByText, fetchByImage, regenerate,
    };
};
