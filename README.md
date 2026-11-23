# Plant Leaf Disease Classifier

Academic project: classify a photographed leaf into one of 38 PlantVillage classes (14 crop
species) and show the matching disease information.

**Scope is frozen** — see [PLAN.md](PLAN.md). Web demo only, local inference, no native app.

**Location:** `C:\plant-doctor`. It lives outside `Documents` on purpose — Windows Defender
Controlled Folder Access blocks script writes there, so downloads, training output and reports
would fail inside the DSH workspace.

```
data/            raw dataset (git-ignored) + processed manifests
models/          exported ONNX
reports/         dataset audit, metrics, figures, project reports
scripts/         download, prepare, audit
src/             data pipeline, training, evaluation, serving
web/             single-page browser interface
knowledge/       disease records (JSON)
```

## Project status

- Dataset extracted: **54,305 color images**, 38 classes, all 256×256, **0 corrupt**.
- Official leaf-aware split: 43,596 train / 10,709 test. A stratified 4,362-image validation set is
  carved out of train; the test set is never touched.
- Duplicate audit: 24 exact-duplicate groups and 558 near-duplicate pairs, **0 crossing a split
  boundary** — the official split does not leak.
- **E1 (linear probe, frozen MobileNetV3-Small): test top-1 97.26%, top-3 99.88%, macro-F1 96.51%**
  → `reports/E1_linear_probe.md`.
- **E2 (fine-tuned MobileNetV3-Small, 10 epochs, RX 9070): test top-1 99.33%, top-3 99.97%,
  macro-F1 99.11%** → `reports/E2_mobilenet_v3_small.md`; ONNX exported with verified parity.
- **GPU:** the AMD Radeon RX 9070 now trains through AMD's ROCm-for-Windows PyTorch
  (`.venv-rocm`, torch 2.9.1+rocm7.2.1). MIOpen's JIT is broken in that SDK, so the script
  disables `torch.backends.cudnn` and uses native HIP kernels.
- **Application implemented:** FastAPI serves the ONNX model and a single-page browser interface
  for camera capture or image upload, top-three predictions, confidence gating, and disease
  information. French progress report 2 is `reports/Rapport_Avancement_2_FR.pdf` and follows the
  first report on dataset preparation and model evaluation.
- **Remaining activity:** prepare and deliver the app demonstration. A field-photo study remains
  optional and has not been completed. E3–E8 were removed from scope.

## Setup

Two environments:

- `.venv` — CPU PyTorch (2.14.1+cpu). Used for E1 and anything not training-critical.
- `.venv-rocm` — PyTorch 2.9.1+rocm7.2.1 for the RX 9070 (installed per
  [AMD's Windows guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)).
  Used for E2 training.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python -m venv .venv-rocm
.venv-rocm\Scripts\python.exe -m pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
.venv-rocm\Scripts\python.exe -m pip install matplotlib onnx
```

## Get the data

The Hugging Face mirror is the default path — no account, and it ships the official leaf-aware
80/20 split (43,596 train / 10,709 test):

```bash
python scripts/download_data.py --source hf --dest data/raw/plantvillage_hf
```

Kaggle works too (needs `~/.kaggle/kaggle.json`), but has no split manifests:

```bash
python scripts/download_data.py --source kaggle
```

## Prepare manifests

```bash
python scripts/prepare_data.py
```

Extracts `data.zip`, validates the official split files, carves a stratified 10% validation set out
of train, and writes `data/processed/{train,val,test}.csv` + `class_index.json` + `split_summary.md`.
The official test set is never touched.

## Audit before training

```bash
python scripts/audit_dataset.py --data-dir data/raw/plantvillage_hf --out-dir reports ^
    --variant color --splits-dir data/processed
```

Writes `reports/dataset_report.md`, `reports/class_counts.csv`, `reports/duplicates.csv`,
`reports/near_duplicates.csv`. `--splits-dir` adds the train/test leakage check.
Do not train until class count is 38 and duplicates are known.

## Run the app

For inference on another computer, clone this repository and install only the serving dependencies:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-serve.txt
```

The final ONNX model and its ordered class-name list are included in the repository. The raw
dataset, processed training manifests and training checkpoints are not needed to run the app.

Start the service:

```bash
.\.venv\Scripts\python.exe -m uvicorn src.serve.app:app --host 127.0.0.1 --port 8010
```

Then open <http://127.0.0.1:8010> on that computer. To use another device on the same local
network, start with `--host 0.0.0.0` and open the host computer's local IP address on the other
device. Camera access may require HTTPS there; image upload remains available.

| Endpoint | Purpose |
|---|---|
| `GET /` | Single-page UI (camera or file upload → top-3 → disease card) |
| `GET /health` | Confirms the model loads and reports the model name and class count |
| `POST /predict` | multipart `file` (JPEG/PNG/WebP) → top-3 predictions, confidence, disease details |

The service loads `models/E2_mobilenet_v3_small.onnx` and
`models/E2_mobilenet_v3_small.classes.json` with ONNX Runtime (no torch or `data/` folder at
inference time). Predictions below 0.60 confidence, or with a margin under 0.15 over the runner-up,
are returned with `"confident": false` and the UI shows "Not sure" instead of a disease name.
Disease text comes from `knowledge/diseases.json`.
