"use client";

import { useRef, useState } from "react";

/**
 * Query-by-image: upload one or more mood images (screenshots, photos,
 * paintings). Multiple images are mean-pooled into one query in the engine.
 */
export const ImageSearch = ({ onSearch, isLoading }) => {
    const [files, setFiles] = useState([]);
    const inputRef = useRef(null);

    const addFiles = (list) => {
        const imgs = Array.from(list || []).filter((f) => f.type.startsWith("image/"));
        if (imgs.length) setFiles((prev) => [...prev, ...imgs].slice(0, 8)); // cap at 8
    };

    const removeAt = (i) => setFiles((prev) => prev.filter((_, k) => k !== i));

    const onDrop = (e) => {
        e.preventDefault();
        addFiles(e.dataTransfer.files);
    };

    return (
        <div className="flex flex-col gap-3">
            <div className="flex flex-col sm:flex-row gap-4 items-stretch">
                <div
                    onClick={() => inputRef.current?.click()}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={onDrop}
                    className={`flex-1 flex items-center justify-center rounded-xl border border-dashed cursor-pointer transition-colors duration-300 min-h-32 p-3 ${
                        files.length ? "border-amber-400/70" : "border-stone-700 hover:border-amber-400/50"
                    }`}
                >
                    {files.length ? (
                        <div className="flex flex-wrap gap-2 w-full">
                            {files.map((f, i) => (
                                <div key={i} className="relative group">
                                    <img
                                        src={URL.createObjectURL(f)}
                                        alt=""
                                        className="h-20 w-20 object-cover rounded-lg border border-stone-700"
                                    />
                                    <button
                                        onClick={(e) => { e.stopPropagation(); removeAt(i); }}
                                        className="absolute -top-1.5 -right-1.5 bg-stone-900 border border-stone-600 text-stone-300 rounded-full w-5 h-5 text-xs leading-none hover:bg-red-900/70"
                                        aria-label="remove"
                                    >×</button>
                                </div>
                            ))}
                            <div className="h-20 w-20 flex items-center justify-center rounded-lg border border-dashed border-stone-700 text-stone-600 text-2xl">+</div>
                        </div>
                    ) : (
                        <span className="text-stone-600 text-sm p-6 text-center">
                            …or drop image(s) that feel right —<br />
                            screenshots, photos, paintings (add several to blend the vibe)
                        </span>
                    )}
                    <input
                        ref={inputRef}
                        type="file"
                        accept="image/*"
                        multiple
                        className="hidden"
                        onChange={(e) => addFiles(e.target.files)}
                    />
                </div>
                <button
                    onClick={() => files.length && onSearch(files)}
                    disabled={isLoading || !files.length}
                    className="px-6 py-3 bg-amber-400 text-stone-950 font-semibold rounded-xl hover:bg-amber-300 transition-all duration-300 disabled:bg-stone-800 disabled:text-stone-600 disabled:cursor-not-allowed self-center whitespace-nowrap"
                >
                    {isLoading ? "searching…" : files.length > 1 ? `Match these ${files.length}` : "Match this image"}
                </button>
            </div>
        </div>
    );
};
