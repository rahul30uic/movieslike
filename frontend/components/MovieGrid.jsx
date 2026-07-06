"use client";

const MovieCard = ({ movie }) => (
    <div className="bg-gray-800 rounded-lg p-4 flex flex-col justify-between items-center border border-gray-700 hover:border-purple-500 transition-all duration-300 hover:scale-[1.02]">
        {movie.poster_path ? (
            <img 
                src={`https://image.tmdb.org/t/p/w500${movie.poster_path}`} 
                alt={movie.title} 
                className="w-full h-auto rounded-md mb-3 object-cover aspect-[2/3] shadow-md" 
            />
        ) : (
            <div className="w-full aspect-[2/3] bg-gray-700 rounded-md flex flex-col items-center justify-center mb-3">
                <span className="text-gray-500 text-sm">No Poster</span>
            </div>
        )}
        <div className="w-full text-center mt-2">
            <h4 className="text-white text-sm font-semibold line-clamp-2 min-h-[40px] flex items-center justify-center">
                {movie.title}
            </h4>
            <p className="text-gray-500 text-xs mt-1 font-mono">TMDB: {movie.tmdb_id}</p>
        </div>
    </div>
);

const SkeletonCard = () => (
    <div className="bg-gray-800 rounded-lg p-4 aspect-[2/3] animate-pulse border border-gray-700">
        <div className="h-full w-full bg-gray-700 rounded"></div>
    </div>
);

export const MovieGrid = ({ movies, isLoading, error }) => {
    if (error) {
        return (
            <div className="text-center py-10 px-4 bg-red-900/20 border border-red-500 rounded-lg">
                <h3 className="text-2xl font-bold text-red-400">An Error Occurred</h3>
                <p className="text-red-300 mt-2 font-mono">{error}</p>
                <p className="text-gray-400 mt-4">
                    Please ensure your FastAPI backend is running at `http://localhost:8000` and that it allows cross-origin requests from this frontend.
                </p>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {Array.from({ length: 12 }).map((_, index) => (
                    <SkeletonCard key={index} />
                ))}
            </div>
        );
    }

    if (movies.length === 0) {
        return (
            <div className="text-center py-10 px-4 bg-gray-800/50 border border-dashed border-gray-600 rounded-lg">
                <h3 className="text-2xl font-semibold text-gray-300">No Recommendations Yet</h3>
                <p className="text-gray-500 mt-2">Select a vibe and click the button to get started.</p>
            </div>
        );
    }

    return (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {movies.map((movie) => (
                <MovieCard key={movie.tmdb_id} movie={movie} />
            ))}
        </div>
    );
};

