"use client";

export const PenaltySlider = ({ alpha, setAlpha }) => {
    return (
        <div className="flex flex-col items-center p-4 bg-gray-800 rounded-lg">
            <div className="w-full flex justify-between text-sm text-gray-400 mb-2">
                <span>More Niche (High Penalty)</span>
                <span>More Popular (Low Penalty)</span>
            </div>
            <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={1 - alpha}
                onChange={(e) => setAlpha(1 - parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
            />
            <div className="mt-2 text-lg font-mono text-white bg-gray-900 px-3 py-1 rounded">Alpha: {alpha.toFixed(2)}</div>
        </div>
    );
};
