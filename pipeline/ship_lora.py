"""
Ship the LoRA fine-tune into the serving path.

1. Merge the best adapter into the base SigLIP vision tower.
2. Re-encode every corpus image with the tuned tower
   -> data/post_modality_vectors_lora.npz (schema matches encode_modalities).
3. Export the merged tower to quantized ONNX in the transformers.js local
   layout -> frontend/public/models/siglip-lora/, so BROWSER image queries
   embed in the same tuned space as the index.

Usage:
    python pipeline/ship_lora.py
"""

import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
ADAPTER_DIR = os.path.join(DATA_DIR, "siglip_lora_best")
MERGED_DIR = os.path.join(DATA_DIR, "siglip_lora_merged")
WEB_MODEL_DIR = os.path.join(REPO_ROOT, "frontend", "public", "models", "siglip-lora")
INPUT_CSV = os.path.join(DATA_DIR, "final_dataset.csv")
NPZ_OUT = os.path.join(DATA_DIR, "post_modality_vectors_lora.npz")

BASE = "google/siglip-base-patch16-224"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 64


def merged_model():
    if os.path.exists(os.path.join(MERGED_DIR, "config.json")):
        logging.info("Loading already-merged tower...")
        return SiglipVisionModel.from_pretrained(MERGED_DIR)
    base = SiglipVisionModel.from_pretrained(BASE)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model = model.merge_and_unload()
    model.save_pretrained(MERGED_DIR)
    logging.info(f"Merged adapter -> {MERGED_DIR}")
    return model


def encode_corpus(model, processor):
    df = pd.read_csv(INPUT_CSV)

    def paths(s):
        if not isinstance(s, str):
            return []
        out = [os.path.join(DATA_DIR, p.strip()) for p in s.split("|") if p.strip()]
        return [p for p in out if os.path.exists(p)]

    df["img_paths"] = df["image_local_path"].apply(paths)
    n = len(df)
    dim = model.config.hidden_size
    image_vecs = np.zeros((n, dim), dtype=np.float32)
    has_image = np.zeros(n, dtype=bool)

    flat, rows = [], []
    for i, ps in enumerate(df["img_paths"]):
        for p in ps:
            flat.append(p)
            rows.append(i)
    logging.info(f"Encoding {len(flat)} images with the tuned tower...")

    flat_vecs = np.zeros((len(flat), dim), dtype=np.float32)
    ok = np.zeros(len(flat), dtype=bool)
    model = model.to(DEVICE).eval()
    with torch.no_grad():
        for s in range(0, len(flat), BATCH):
            chunk = flat[s:s + BATCH]
            images, keep = [], []
            for j, p in enumerate(chunk):
                try:
                    images.append(Image.open(p).convert("RGB"))
                    keep.append(s + j)
                except Exception:
                    pass
            if not images:
                continue
            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            out = model(**inputs).pooler_output
            out = torch.nn.functional.normalize(out, dim=-1)
            flat_vecs[keep] = out.float().cpu().numpy()
            ok[keep] = True
            if (s // BATCH) % 20 == 0:
                logging.info(f"  batch {s // BATCH + 1}/{(len(flat) + BATCH - 1) // BATCH}")

    for i in range(n):
        rws = [k for k, r in enumerate(rows) if r == i and ok[k]]
        if rws:
            avg = flat_vecs[rws].mean(axis=0)
            nrm = np.linalg.norm(avg)
            if nrm > 0:
                image_vecs[i] = avg / nrm
                has_image[i] = True

    np.savez_compressed(NPZ_OUT, image_vecs=image_vecs, has_image=has_image)
    logging.info(f"Saved tuned image vectors: {has_image.sum()} posts -> {NPZ_OUT}")


class VisionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state, out.pooler_output


def export_onnx(model, processor):
    os.makedirs(os.path.join(WEB_MODEL_DIR, "onnx"), exist_ok=True)
    model = model.to("cpu").eval()
    wrapper = VisionWrapper(model)
    dummy = torch.zeros(1, 3, 224, 224)
    fp32_path = os.path.join(WEB_MODEL_DIR, "onnx", "model.onnx")
    torch.onnx.export(
        wrapper, (dummy,), fp32_path,
        input_names=["pixel_values"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={"pixel_values": {0: "batch"},
                      "last_hidden_state": {0: "batch"},
                      "pooler_output": {0: "batch"}},
        opset_version=17,
    )
    logging.info("Exported fp32 ONNX; quantizing to q8...")
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q_path = os.path.join(WEB_MODEL_DIR, "onnx", "model_quantized.onnx")
    quantize_dynamic(fp32_path, q_path, weight_type=QuantType.QUInt8)
    os.remove(fp32_path)  # too big for git; browser uses q8

    # transformers.js local layout: config + preprocessor at the model root
    cfg = json.loads(model.config.to_json_string())
    cfg["model_type"] = "siglip_vision_model"
    with open(os.path.join(WEB_MODEL_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    processor.save_pretrained(WEB_MODEL_DIR)
    size = os.path.getsize(q_path) / 1e6
    logging.info(f"Web model ready: {q_path} ({size:.0f}MB)")


def main():
    processor = AutoImageProcessor.from_pretrained(BASE)
    model = merged_model()
    if os.path.exists(NPZ_OUT):
        logging.info(f"{NPZ_OUT} exists — skipping corpus re-encode.")
    else:
        encode_corpus(model, processor)
    export_onnx(model, processor)
    logging.info("ship_lora complete.")


if __name__ == "__main__":
    main()
