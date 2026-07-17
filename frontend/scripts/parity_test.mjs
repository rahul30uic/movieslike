// Verifies transformers.js (quantized) embeddings match the Python models
// that built the movie index. Run: node scripts/parity_test.mjs
import { pipeline, AutoProcessor, SiglipVisionModel, RawImage } from "@huggingface/transformers";
import { readFileSync } from "fs";

const ref = JSON.parse(readFileSync("/tmp/parity_ref.json", "utf8"));

const cos = (a, b) => {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / (Math.sqrt(na) * Math.sqrt(nb));
};

// --- text ---
const pipe = await pipeline("feature-extraction", "Xenova/bge-base-en-v1.5", { dtype: "q8" });
const out = await pipe(
    "Represent this sentence for searching relevant passages: cozy rainy night, gentle loneliness",
    { pooling: "cls", normalize: true },
);
console.log("text cosine (JS q8 vs Python fp32):", cos(Array.from(out.data), ref.text).toFixed(4));

// --- image ---
const processor = await AutoProcessor.from_pretrained("Xenova/siglip-base-patch16-224");
const vision = await SiglipVisionModel.from_pretrained("Xenova/siglip-base-patch16-224", { dtype: "q8" });
const image = await RawImage.read(ref.img_path);
const inputs = await processor(image);
const vout = await vision(inputs);
console.log("image cosine (JS q8 vs Python fp32):", cos(Array.from(vout.pooler_output.data), ref.image).toFixed(4));
