"use client";

import { useEffect, useRef } from "react";

/**
 * Model introspection overlay: renders the patch-similarity heatmap on top
 * of the uploaded image — brighter amber = that region drove the match.
 */
export const MatchExplanation = ({ explanation }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        if (!explanation) return;
        const { imageUrl, grid, cols, rows } = explanation;
        const img = new Image();
        img.onload = () => {
            const canvas = canvasRef.current;
            if (!canvas) return;
            const maxW = 480;
            const scale = Math.min(maxW / img.width, 1);
            canvas.width = img.width * scale;
            canvas.height = img.height * scale;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            // Small heat canvas, scaled up with smoothing = soft blobs
            const heat = document.createElement("canvas");
            heat.width = cols;
            heat.height = rows;
            const hctx = heat.getContext("2d");
            const data = hctx.createImageData(cols, rows);
            for (let i = 0; i < grid.length; i++) {
                data.data[i * 4] = 251;      // amber
                data.data[i * 4 + 1] = 191;
                data.data[i * 4 + 2] = 36;
                data.data[i * 4 + 3] = Math.round(grid[i] * 190);
            }
            hctx.putImageData(data, 0, 0);
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = "high";
            ctx.drawImage(heat, 0, 0, canvas.width, canvas.height);
        };
        img.src = explanation.imageUrl;
    }, [explanation]);

    if (!explanation) return null;

    return (
        <div className="flex flex-col items-center mb-10">
            <canvas ref={canvasRef} className="rounded-xl border border-stone-800 shadow-xl shadow-black/40 max-w-full" />
            <p className="text-stone-500 text-xs mt-3 max-w-md text-center">
                <span className="text-amber-300/90">Model introspection:</span> brighter regions are
                the parts of your image that drove the match with{" "}
                <span className="font-display italic text-stone-400">{explanation.vs}</span>{" "}
                (SigLIP patch tokens vs. the movie&apos;s image-block vector).
            </p>
        </div>
    );
};
