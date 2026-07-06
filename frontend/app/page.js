"use client";

import { VibeAnchors } from "@/components/VibeAnchors";
import { PenaltySlider } from "@/components/PenaltySlider";
import { MovieGrid } from "@/components/MovieGrid";
import { useRecommendations } from "@/hooks/useRecommendations";

export default function Home() {
  const {
    selectedAnchor,
    setSelectedAnchor,
    alpha,
    setAlpha,
    movies,
    isLoading,
    error,
    fetchRecommendations
  } = useRecommendations();

  return (
    <main className="min-h-screen bg-gray-900 text-white p-4 sm:p-8">
      <div className="max-w-7xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-4xl sm:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500">
            MovieVibes AI
          </h1>
          <p className="text-gray-400 mt-2">Select a vibe, tune the popularity, and discover your next watch.</p>
        </header>

        <section className="mb-12">
          <h2 className="text-2xl font-semibold mb-4 border-b border-gray-700 pb-2">1. Choose a Vibe Profile</h2>
          <VibeAnchors selectedAnchor={selectedAnchor} onSelectAnchor={setSelectedAnchor} />
        </section>

        <section className="mb-12 max-w-2xl mx-auto">
          <h2 className="text-2xl font-semibold mb-4 border-b border-gray-700 pb-2">2. Adjust Popularity Bias (Alpha)</h2>
          <PenaltySlider alpha={alpha} setAlpha={setAlpha} />
        </section>

        <div className="text-center mb-12">
          <button
            onClick={fetchRecommendations}
            disabled={isLoading || !selectedAnchor}
            className="px-8 py-3 bg-purple-600 text-white font-bold rounded-lg shadow-lg hover:bg-purple-700 transition-all duration-300 disabled:bg-gray-600 disabled:cursor-not-allowed disabled:shadow-none"
          >
            {isLoading ? "Discovering..." : "Get Recommendations"}
          </button>
        </div>

        <section>
          <MovieGrid movies={movies} isLoading={isLoading} error={error} />
        </section>
      </div>
    </main>
  );
}
