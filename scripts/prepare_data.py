#!/usr/bin/env python
"""Prepare the dataset: extract the archive, validate the official splits, and write manifests.

The Hugging Face mirror ships leaf-aware 80/20 splits (43,596 train / 10,709 test). We keep the
official test set untouched, carve a stratified validation split out of train, and freeze all of it
into CSV manifests so every experiment uses identical data.

    python scripts/prepare_data.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

SPLIT_FILES = {
    "train": "color_train.txt",
    "test": "color_test.txt",
}


def extract(raw_dir: Path, force: bool = False) -> Path:
    """Extract data.zip once; returns the directory that holds the class folders."""
    zip_path = raw_dir / "data.zip"
    inner = raw_dir / "raw"
    if inner.is_dir() and not force:
        print(f"[extract] already present: {inner}")
        return inner
    if not zip_path.is_file():
        raise SystemExit(f"missing archive: {zip_path} (run scripts/download_data.py)")
    print(f"[extract] {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB) -> {raw_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"[extract] {len(names):,} entries")
        for i, name in enumerate(names, 1):
            zf.extract(name, raw_dir)
            if i % 20000 == 0:
                print(f"  {i:,}/{len(names):,}")
    return inner


def read_split(raw_dir: Path, name: str) -> list[str]:
    path = raw_dir / "splits" / SPLIT_FILES[name]
    if not path.is_file():
        raise SystemExit(f"missing split file: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines


def label_of(rel_path: str) -> str:
    return Path(rel_path).parent.name


def stratified_split(paths: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    by_class: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        by_class[label_of(p)].append(p)
    rng = random.Random(seed)
    train, val = [], []
    for cls in sorted(by_class):
        items = sorted(by_class[cls])
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/plantvillage_hf"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()

    raw_dir = args.raw_dir.resolve()
    inner = extract(raw_dir, force=args.force_extract)

    train_all = read_split(raw_dir, "train")
    test = read_split(raw_dir, "test")
    print(f"[splits] official train={len(train_all):,} test={len(test):,}")

    missing = [p for p in train_all + test if not (raw_dir / p).is_file()]
    if missing:
        print(f"[validate] WARNING: {len(missing):,} listed files do not exist, e.g. {missing[:3]}")

    overlap = set(train_all) & set(test)
    if overlap:
        print(f"[validate] WARNING: {len(overlap):,} paths appear in both splits")

    train, val = stratified_split(train_all, args.val_frac, args.seed)
    splits = {"train": train, "val": val, "test": test}
    labels = sorted({label_of(p) for p in train_all + test})
    label_to_idx = {label: i for i, label in enumerate(labels)}

    if len(labels) != 38:
        print(f"[validate] WARNING: found {len(labels)} classes, expected 38")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, paths in splits.items():
        out = args.out_dir / f"{name}.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "label_idx", "crop", "disease", "split"])
            for p in sorted(paths):
                label = label_of(p)
                crop, _, disease = label.partition("___")
                writer.writerow([p, label, label_to_idx[label], crop, disease or "healthy", name])
        print(f"[write] {out} ({len(paths):,} rows)")

    (args.out_dir / "class_index.json").write_text(
        json.dumps(label_to_idx, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # quick summary for the report
    lines = ["# Split summary", "", "| class | train | val | test |", "|---|---:|---:|---:|"]
    for label in labels:
        row = [sum(1 for p in splits[s] if label_of(p) == label) for s in ("train", "val", "test")]
        lines.append(f"| {label} | {row[0]:,} | {row[1]:,} | {row[2]:,} |")
    totals = [len(splits[s]) for s in ("train", "val", "test")]
    lines.append(f"| **total** | **{totals[0]:,}** | **{totals[1]:,}** | **{totals[2]:,}** |")
    (args.out_dir / "split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] {len(labels)} classes | train={len(train):,} val={len(val):,} test={len(test):,}")
    counts = Counter(label_of(p) for p in train_all)
    rare = counts.most_common()[-3:]
    print(f"[balance] rarest train classes: {rare}")


if __name__ == "__main__":
    main()
