"use client";

import Link from "next/link";
import { ProbeFlow } from "@/components/ProbeFlow";
import { VibeSearch } from "@/components/VibeSearch";
import { ImageSearch } from "@/components/ImageSearch";
import { PenaltySlider } from "@/components/PenaltySlider";
import { MovieGrid } from "@/components/MovieGrid";
import { MatchExplanation } from "@/components/MatchExplanation";
import { useRecommendations } from "@/hooks/useRecommendations";

export default function Home() {
  const {
    alpha,
    setAlpha,
    movies,
    isLoading,
    error,
    engineStatus,
    explanation,
    hasLastSearch,
    fetchByText,
    fetchByImage,
    regenerate
  } = useRecommendations();

  return (
    <main className="min-h-screen p-4 sm:p-8 vignette">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mt-6 mb-10">
          <h1 className="font-display text-5xl sm:text-6xl text-stone-100 tracking-tight">
            Movieslike
          </h1>
          <p className="text-stone-400 mt-4 text-lg">
            Stop scrolling. <span className="font-display italic text-amber-200">Which of these feels more like tonight?</span>
          </p>
        </header>

        <ProbeFlow />

        <div className="flex items-center gap-4 mt-16 mb-10">
          <div className="flex-1 h-px bg-stone-800" />
          <span className="font-display italic text-stone-500">or tell us yourself</span>
          <div className="flex-1 h-px bg-stone-800" />
        </div>

        <section className="mb-8 max-w-3xl mx-auto">
          <VibeSearch onSearch={fetchByText} isLoading={isLoading} />
        </section>

        <section className="mb-8 max-w-3xl mx-auto">
          <ImageSearch onSearch={fetchByImage} isLoading={isLoading} />
        </section>

        <section className="mb-8 max-w-2xl mx-auto">
          <PenaltySlider alpha={alpha} setAlpha={setAlpha} />
          {hasLastSearch && (
            <div className="text-center mt-4">
              <button
                onClick={regenerate}
                disabled={isLoading}
                className="px-6 py-2.5 border border-amber-400/60 text-amber-200 rounded-full hover:bg-amber-400/10 transition-colors duration-300 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {isLoading ? "regenerating…" : "↻ regenerate with these settings"}
              </button>
            </div>
          )}
        </section>

        {engineStatus && (
          <p className="text-center text-amber-200/90 text-sm mb-6 animate-pulse">{engineStatus}</p>
        )}

        <section className="mb-8">
          <MovieGrid movies={movies} isLoading={isLoading} error={error} />
        </section>

        <MatchExplanation explanation={explanation} />

        <p className="text-center text-stone-500 text-sm mt-6">
          Curious how the space is organized?{" "}
          <Link href="/atlas" className="text-amber-300/90 hover:text-amber-200 underline underline-offset-4 decoration-stone-700">
            explore the vibe atlas →
          </Link>
        </p>

        <p className="text-center text-stone-700 text-xs mt-10 mb-4">
          Vibes learned from 4,400 Reddit &ldquo;movies that feel like this?&rdquo; posts ·{" "}
          <a href="https://github.com/rahul30uic/movieslike" className="hover:text-stone-500 underline underline-offset-2">
            how it works
          </a>
          <br />
          This product uses TMDB and the TMDB APIs but is not endorsed, certified, or otherwise approved by TMDB.
          <br />
          Mood images originate from public Reddit posts; rights remain with their owners —
          contact for removal.
        </p>
      </div>
    </main>
  );
}
