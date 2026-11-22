"""FastAPI service for the leaf disease classifier.

Serves the static single-page UI and a /predict endpoint backed by the ONNX model.
No torch at inference time - only onnxruntime, numpy and Pillow.

    python -m uvicorn src.serve.app:app --host 127.0.0.1 --port 8010
"""
from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "E2_mobilenet_v3_small.onnx"
INDEX = ROOT / "data" / "processed" / "class_index.json"
KNOWLEDGE = ROOT / "knowledge" / "diseases.json"
WEB = ROOT / "web" / "index.html"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIDENCE_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.15
MAX_BYTES = 10_000_000

app = FastAPI(title="Leaf Disease Classifier", version="2.0")


@lru_cache
def load_model() -> ort.InferenceSession:
    if not MODEL.is_file():
        raise RuntimeError(f"model not found: {MODEL}")
    return ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])


@lru_cache
def load_classes() -> list[str]:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(index.items(), key=lambda kv: kv[1])]


@lru_cache
def load_knowledge() -> dict:
    return json.loads(KNOWLEDGE.read_text(encoding="utf-8"))


def preprocess(data: bytes) -> np.ndarray:
    """Resize short side to 256, centre-crop 224, normalise with ImageNet statistics."""
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w < 32 or h < 32:
            raise ValueError("image too small")
        scale = 256 / min(w, h)
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                       Image.Resampling.BILINEAR)
        w, h = im.size
        left, top = (w - 224) // 2, (h - 224) // 2
        im = im.crop((left, top, left + 224, top + 224))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.transpose(arr, (2, 0, 1))[None].astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB)


@app.get("/health")
def health() -> dict:
    try:
        load_model()
        classes = load_classes()
    except Exception as exc:  # noqa: BLE001 - expose unavailable inference dependency
        raise HTTPException(status_code=503, detail=f"model unavailable: {exc}") from exc
    return {"status": "ok", "model": MODEL.name, "classes": len(classes)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/jpg"}:
        raise HTTPException(status_code=415, detail="upload a JPEG, PNG or WebP image")

    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="image larger than 10 MB")

    try:
        x = preprocess(data)
    except Exception:  # noqa: BLE001 - any decode failure means a bad upload
        raise HTTPException(status_code=400, detail="could not decode image")

    session = load_model()
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: x})[0]
    probs = softmax(logits)[0]

    classes = load_classes()
    knowledge = load_knowledge()
    top = []
    for i in np.argsort(-probs)[:3]:
        label = classes[int(i)]
        info = knowledge.get(label, {})
        top.append({
            "label": label,
            "confidence": round(float(probs[int(i)]), 4),
            "crop": info.get("crop", label.split("___")[0]),
            "disease": info.get("disease", label.split("___")[-1].replace("_", " ")),
            "healthy": info.get("disease") == "Healthy",
        })

    best = top[0]
    margin = best["confidence"] - top[1]["confidence"]
    confident = best["confidence"] >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD

    return JSONResponse({
        "predictions": top,
        "confident": confident,
        "message": None if confident else
        "Low confidence - try a closer, well-lit photo of a single leaf.",
        "details": knowledge.get(best["label"], {}),
        "disclaimer": "Screening aid only - not a substitute for a professional diagnosis.",
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serve.app:app", host="127.0.0.1", port=8010)
