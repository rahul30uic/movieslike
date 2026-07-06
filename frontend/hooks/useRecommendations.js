"use client";

import { useState, useCallback } from 'react';

const API_BASE_URL = "http://localhost:8000";

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

    /**
     * Fetches movie recommendations from the FastAPI backend.
     * This function is explicitly triggered by the UI.
     */
    const fetchRecommendations = useCallback(async () => {
        if (!selectedAnchor) {
            setError("Please select a vibe anchor first.");
            return;
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

    return { selectedAnchor, setSelectedAnchor, alpha, setAlpha, movies, isLoading, error, fetchRecommendations };
};
