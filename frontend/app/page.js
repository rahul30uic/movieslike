"use client";

import { VibeAnchors } from "@/components/VibeAnchors";
import { VibeSearch } from "@/components/VibeSearch";
import { ImageSearch } from "@/components/ImageSearch";
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
    engineStatus,
    fetchRecommendations,
    fetchByText,
    fetchByImage
  } = useRecommendations();

  return (
    <main className="min-h-screen bg-gray-900 text-white p-4 sm:p-8">
      <div className="max-w-7xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-4xl sm:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500">
            MovieVibes AI
          </h1>
          <p className="text-gray-400 mt-2">Select a vibe, tune the popularity, and discover your next watch.</p>
          {process.env.NEXT_PUBLIC_ENABLE_PROBE !== "false" && (
            <a
              href="/probe"
              className="inline-block mt-5 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-bold rounded-lg shadow-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-300"
            >
              Can&apos;t put it in words? Lock in your head-space →
            </a>
          )}
        </header>

        <section className="mb-12">
          <h2 className="text-2xl font-semibold mb-4 border-b border-gray-700 pb-2">1. Choose a Vibe Profile</h2>
          <VibeAnchors selectedAnchor={selectedAnchor} onSelectAnchor={setSelectedAnchor} />
        </section>

        <section className="mb-12 max-w-3xl mx-auto">
          <h2 className="text-2xl font-semibold mb-4 border-b border-gray-700 pb-2">…or describe the vibe yourself</h2>
          <VibeSearch onSearch={fetchByText} isLoading={isLoading} />
        </section>

        <section className="mb-12 max-w-3xl mx-auto">
          <h2 className="text-2xl font-semibold mb-4 border-b border-gray-700 pb-2">…or show us an image that feels right</h2>
          <ImageSearch onSearch={fetchByImage} isLoading={isLoading} />
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

        {engineStatus && (
          <p className="text-center text-purple-300 text-sm mb-6 animate-pulse">{engineStatus}</p>
        )}

        <section>
          <MovieGrid movies={movies} isLoading={isLoading} error={error} />
        </section>
      </div>
    </main>
  );
}
