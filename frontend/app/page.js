"use client";

import { VibeSearch } from "@/components/VibeSearch";
import { ImageSearch } from "@/components/ImageSearch";
import { PenaltySlider } from "@/components/PenaltySlider";
import { MovieGrid } from "@/components/MovieGrid";
import { useRecommendations } from "@/hooks/useRecommendations";

export default function Home() {
  const {
    alpha,
    setAlpha,
    movies,
    isLoading,
    error,
    engineStatus,
    fetchByText,
    fetchByImage
  } = useRecommendations();

  return (
    <main className="min-h-screen bg-gray-900 text-white p-4 sm:p-8">
      <div className="max-w-7xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-4xl sm:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500">
            Movieslike
          </h1>
          <p className="text-gray-400 mt-2">
            Describe a mood — in words or with an image — and get movies that feel like it.
          </p>
          <a
            href="/probe"
            className="inline-block mt-5 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-bold rounded-lg shadow-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-300"
          >
            Can&apos;t put it in words? Lock in your head-space →
          </a>
        </header>

        <section className="mb-12 max-w-3xl mx-auto">
          <h2 className="text-2xl font-semibold mb-4 border-b border-gray-700 pb-2">Describe the vibe</h2>
          <VibeSearch onSearch={fetchByText} isLoading={isLoading} />
        </section>

        <section className="mb-12 max-w-3xl mx-auto">
          <h2 className="text-2xl font-semibold mb-4 border-b border-gray-700 pb-2">…or show us an image that feels right</h2>
          <ImageSearch onSearch={fetchByImage} isLoading={isLoading} />
        </section>

        <section className="mb-12 max-w-2xl mx-auto">
          <h2 className="text-2xl font-semibold mb-4 border-b border-gray-700 pb-2">Hidden gems ↔ crowd favorites</h2>
          <PenaltySlider alpha={alpha} setAlpha={setAlpha} />
        </section>

        {engineStatus && (
          <p className="text-center text-purple-300 text-sm mb-6 animate-pulse">{engineStatus}</p>
        )}

        <section>
          <MovieGrid movies={movies} isLoading={isLoading} error={error} />
        </section>

        <p className="text-center text-gray-600 text-xs mt-12">
          This product uses TMDB and the TMDB APIs but is not endorsed, certified,
          or otherwise approved by TMDB.
        </p>
      </div>
    </main>
  );
}
