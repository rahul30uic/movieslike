"use client";

import Link from "next/link";
import { VibeSearch } from "@/components/VibeSearch";
import { ImageSearch } from "@/components/ImageSearch";
import { PenaltySlider } from "@/components/PenaltySlider";
import { MovieGrid } from "@/components/MovieGrid";
import { MatchExplanation } from "@/components/MatchExplanation";
import { useRecommendations } from "@/hooks/useRecommendations";

export default function SearchPage() {
  const {
    alpha,
    setAlpha,
    movies,
    isLoading,
    error,
    engineStatus,
    explanation,
    fetchByText,
    fetchByImage
  } = useRecommendations();

  return (
    <main className="min-h-screen p-4 sm:p-8 vignette">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mt-6 mb-10">
          <h1 className="font-display text-4xl sm:text-5xl text-stone-100 tracking-tight">
            <Link href="/">Movieslike</Link>
          </h1>
          <p className="text-stone-400 mt-3">
            Describe tonight&apos;s mood — or show us an image that feels right.
          </p>
          <Link
            href="/"
            className="inline-block mt-3 text-sm text-amber-300/90 hover:text-amber-200 underline underline-offset-4 decoration-stone-700 hover:decoration-amber-300 transition-colors"
          >
            ← or just react to images instead
          </Link>
        </header>

        <section className="mb-10 max-w-3xl mx-auto">
          <VibeSearch onSearch={fetchByText} isLoading={isLoading} />
        </section>

        <section className="mb-10 max-w-3xl mx-auto">
          <ImageSearch onSearch={fetchByImage} isLoading={isLoading} />
        </section>

        <section className="mb-10 max-w-2xl mx-auto">
          <PenaltySlider alpha={alpha} setAlpha={setAlpha} />
        </section>

        {engineStatus && (
          <p className="text-center text-amber-200/90 text-sm mb-6 animate-pulse">{engineStatus}</p>
        )}

        <section>
          <MovieGrid movies={movies} isLoading={isLoading} error={error} />
        </section>

        <MatchExplanation explanation={explanation} />

        <p className="text-center text-stone-500 text-sm mt-4">
          Curious how the space is organized?{" "}
          <Link href="/atlas" className="text-amber-300/90 hover:text-amber-200 underline underline-offset-4 decoration-stone-700">
            explore the vibe atlas →
          </Link>
        </p>

        <p className="text-center text-stone-700 text-xs mt-12 mb-4">
          This product uses TMDB and the TMDB APIs but is not endorsed, certified,
          or otherwise approved by TMDB.
        </p>
      </div>
    </main>
  );
}
