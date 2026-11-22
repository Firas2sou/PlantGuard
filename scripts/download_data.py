#!/usr/bin/env python
"""Fetch the PlantVillage dataset from Kaggle or the Hugging Face mirror.

    python scripts/download_data.py --source kaggle
    python scripts/download_data.py --source hf --dest data/raw/hf
    python scripts/download_data.py --source auto          # kaggle, fall back to hf

Kaggle needs ~/.kaggle/kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

KAGGLE_SLUG = "abdallahalidev/plantvillage-dataset"
HF_REPO = "mohanty/PlantVillage"


def via_kaggle() -> Path:
    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub is not installed: pip install kagglehub")
    print(f"[kaggle] downloading {KAGGLE_SLUG} ...")
    return Path(kagglehub.dataset_download(KAGGLE_SLUG))


def via_hf(dest: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is not installed: pip install huggingface_hub")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[hf] downloading {HF_REPO} -> {dest} ...")
    return Path(
        snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=str(dest))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["auto", "kaggle", "hf"], default="auto")
    ap.add_argument("--dest", type=Path, default=Path("data/raw/hf"),
                    help="destination for the Hugging Face mirror")
    args = ap.parse_args()

    path: Path | None = None
    if args.source in ("auto", "kaggle"):
        try:
            path = via_kaggle()
        except Exception as exc:  # noqa: BLE001 - fall back on any kaggle failure
            if args.source == "kaggle":
                sys.exit(f"[kaggle] failed: {exc}")
            print(f"[kaggle] unavailable ({exc}); falling back to Hugging Face")

    if path is None:
        path = via_hf(args.dest)

    print(f"\nDataset root: {path}")
    top = sorted(p.name for p in path.iterdir())
    print(f"Top-level entries ({len(top)}): {top[:12]}{' ...' if len(top) > 12 else ''}")
    print("\nNext: python scripts/audit_dataset.py --data-dir \"<root>\"")


if __name__ == "__main__":
    main()
