"use client";

import { useRef, useState } from "react";

/**
 * Query-by-image: upload any mood image (screenshot, photo, painting) and
 * search the image block of the hybrid embedding space.
 */
export const ImageSearch = ({ onSearch, isLoading }) => {
    const [preview, setPreview] = useState(null);
    const [file, setFile] = useState(null);
    const inputRef = useRef(null);

    const handleFile = (f) => {
        if (!f || !f.type.startsWith("image/")) return;
        setFile(f);
        setPreview(URL.createObjectURL(f));
    };

    const onDrop = (e) => {
        e.preventDefault();
        handleFile(e.dataTransfer.files?.[0]);
    };

    return (
        <div className="flex flex-col sm:flex-row gap-4 items-stretch">
            <div
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={onDrop}
                className={`flex-1 flex items-center justify-center rounded-xl border border-dashed cursor-pointer transition-colors duration-300 min-h-32 overflow-hidden ${
                    preview ? "border-amber-400/70" : "border-stone-700 hover:border-amber-400/50"
                }`}
            >
                {preview ? (
                    <img src={preview} alt="your mood" className="max-h-48 w-full object-cover" />
                ) : (
                    <span className="text-stone-600 text-sm p-6 text-center">
                        …or drop an image that feels right —<br />
                        a screenshot, a photo you took, a painting
                    </span>
                )}
                <input
                    ref={inputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => handleFile(e.target.files?.[0])}
                />
            </div>
            <button
                onClick={() => file && onSearch(file)}
                disabled={isLoading || !file}
                className="px-6 py-3 bg-amber-400 text-stone-950 font-semibold rounded-xl hover:bg-amber-300 transition-all duration-300 disabled:bg-stone-800 disabled:text-stone-600 disabled:cursor-not-allowed self-center whitespace-nowrap"
            >
                {isLoading ? "searching…" : "Match this image"}
            </button>
        </div>
    );
};
