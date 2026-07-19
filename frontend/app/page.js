"use client";

import Link from "next/link";
import { ProbeFlow } from "@/components/ProbeFlow";

export default function Home() {
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

        <div className="text-center mt-14 mb-8">
          <p className="text-stone-500 text-sm">
            Rather say it than feel it out?{" "}
            <Link href="/search" className="text-amber-300/90 hover:text-amber-200 underline underline-offset-4 decoration-stone-700 hover:decoration-amber-300 transition-colors">
              describe the mood in words, or upload an image →
            </Link>
          </p>
          <p className="text-stone-600 text-sm mt-2">
            or{" "}
            <Link href="/atlas" className="hover:text-stone-400 underline underline-offset-4 decoration-stone-800">
              wander the map of every mood we know
            </Link>
          </p>
        </div>

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
