"use client";

const MovieCard = ({ movie }) => (
    <div className="group rounded-xl overflow-hidden border border-stone-800 hover:border-amber-400/60 bg-stone-950 transition-all duration-300 hover:scale-[1.02] shadow-xl shadow-black/40">
        {movie.poster_path ? (
            <img
                src={`https://image.tmdb.org/t/p/w500${movie.poster_path}`}
                alt={movie.title}
                className="w-full h-auto object-cover aspect-[2/3] saturate-[0.95] group-hover:saturate-105 transition-all duration-500"
            />
        ) : (
            <div className="w-full aspect-[2/3] bg-stone-900 flex items-center justify-center">
                <span className="text-stone-600 text-sm font-display italic">no poster</span>
            </div>
        )}
        <div className="p-3 text-center">
            <h4 className="text-stone-200 text-sm font-medium line-clamp-2 min-h-[40px] flex items-center justify-center">
                {movie.title}
            </h4>
        </div>
    </div>
);

const SkeletonCard = () => (
    <div className="rounded-xl border border-stone-800 bg-stone-900/60 aspect-[2/3] animate-pulse" />
);

export const MovieGrid = ({ movies, isLoading, error }) => {
    if (error) {
        return (
            <div className="text-center py-10 px-4 border border-red-900/60 bg-red-950/20 rounded-xl">
                <h3 className="font-display text-xl text-red-300">Something went wrong</h3>
                <p className="text-red-300/70 mt-2 text-sm">{error}</p>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
                {Array.from({ length: 5 }).map((_, index) => (
                    <SkeletonCard key={index} />
                ))}
            </div>
        );
    }

    if (!movies || movies.length === 0) {
        return null;
    }

    return (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
            {movies.map((movie) => (
                <MovieCard key={movie.tmdb_id} movie={movie} />
            ))}
        </div>
    );
};
