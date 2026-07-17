"use client";

export const PenaltySlider = ({ alpha, setAlpha }) => {
    return (
        <div className="flex flex-col items-center px-5 py-4 rounded-xl border border-stone-800 bg-stone-950/60">
            <div className="w-full flex justify-between text-xs uppercase tracking-widest text-stone-500 mb-3">
                <span>crowd favorites</span>
                <span>hidden gems</span>
            </div>
            <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-stone-800 rounded-lg appearance-none cursor-pointer accent-amber-400"
            />
        </div>
    );
};
