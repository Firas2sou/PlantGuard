# Plant Leaf Disease Classifier — Academic Project Plan (v2, simplified)

**Purpose:** an academic deliverable — a prepared dataset, one strong lightweight model, a simple
web demo, and two sequential progress reports. Everything non-essential from v1 has been cut.

**Frozen scope:** 38 classes / 14 crops, PlantVillage color variant, English + French reports,
online inference, single leaf per photo, no offline mode, no native app.

---

## What changed vs v1

| v1 item | v2 decision |
|---|---|
| E1 — linear probe (frozen features) | **Kept** — done, it is the baseline |
| E2 — MobileNetV3-Small fine-tune | **Kept** — done, it is the main model |
| E3 — strong augmentation study | Removed |
| E4 — random-background training | Removed |
| E5 — EfficientNet-B0 | Removed |
| E6 — segmented-vs-color ablation | Removed |
| E7 — real-photo evaluation | Reduced to an **optional informal test** on ~20 photos |
| E8 — calibration study | Reduced to a **simple confidence threshold** in the app |
| React/Vite PWA + build toolchain, Docker | Replaced by **one FastAPI server + one static HTML page** (no Node, no build step) |
| 12-day plan | ~3 days remaining |

Reason: the dataset preparation and the two-model comparison already give a defensible story for an
academic project; the extra experiments mostly repeat the same conclusion.

---

## Models used (the report names these explicitly)

Both models use **MobileNetV3-Small** (Howard et al., 2019), a lightweight convolutional network
designed for mobile devices, pretrained on ImageNet. Input: 224×224.

| # | Model | Role | Trainable params | Test top-1 | Test top-3 | Test macro-F1 |
|---|---|---|---|---|---:|---:|---:|
| E1 | Linear probe on frozen MobileNetV3-Small features (logistic regression) | Baseline | 21,926 | 97.26 % | 99.88 % | 96.51 % |
| E2 | MobileNetV3-Small fine-tuned end-to-end (10 epochs, Radeon RX 9070) | **Main model** | 1,556,806 | **99.33 %** | 99.97 % | 99.11 % |

The main model is exported to ONNX (opset 17) with verified numerical parity, and it is what the
app serves.

---

## Dataset (done)

- PlantVillage, Hugging Face mirror, color variant: 54,305 images, 38 classes, 14 crops, 256×256.
- Official leaf-aware split: 43,596 train / 10,709 test → our split: 39,234 / 4,362 / 10,709.
- Audit: 0 corrupt, 24 exact duplicate groups, 558 near duplicates, **0 crossing split boundaries**.

## App (one file, no toolchain)

- `src/serve/app.py` — FastAPI: `GET /` (static HTML page), `GET /health` (model availability),
  and `POST /predict` (image → top-3 + confidence via ONNX Runtime).
- `web/index.html` — camera or image-file input → preview, top-3 bars and disease info card; a
  confidence threshold shows "not sure" for uncertain results.
- `knowledge/diseases.json` — 38 short records (disease, cause, simple treatment).

## Deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | Prepared dataset + audit report | done |
| 2 | Trained ONNX model (E2) | done |
| 3 | Simple web app (FastAPI + static page) | done |
| 4 | Progress report 1 FR (dataset preparation and model evaluation) | done |
| 5 | Progress report 2 FR (model integration and web application) | done |
| 6 | Optional informal field test (~20 photos) | optional |

## Demonstration

Run the local FastAPI server, open `http://127.0.0.1:8010`, and demonstrate the camera or file
upload flow through the prediction and disease information card. Progress report 2 follows report
1 and records the model integration, implemented app, and demo preparation. A field-photo
evaluation and public deployment are optional extensions, not project completion requirements.

## Reproducibility

```text
python scripts/download_data.py --source hf --dest data/raw/plantvillage_hf
python scripts/prepare_data.py
python scripts/audit_dataset.py --data-dir data/raw/plantvillage_hf --out-dir reports --variant color --splits-dir data/processed
python scripts/e1_linear_probe.py
.venv-rocm\Scripts\python.exe scripts/e2_finetune.py --epochs 10 --batch-size 128 --workers 12
```
