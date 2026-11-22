#!/usr/bin/env python
"""Audit a PlantVillage-style image tree before training.

Reports, per variant (color / grayscale / segmented) and per class:
  * image counts, image dimensions
  * corrupt / unreadable files
  * exact and near-duplicate images (dHash), which cause train/test leakage

Only Pillow is required, so this runs before torch is installed.

    python scripts/audit_dataset.py --data-dir "data/raw/plantvillage dataset"
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png"}
VARIANTS = {"color", "grayscale", "segmented"}

# Canonical classes as named in this release (the HF mirror / current PlantVillage naming).
# Compared after normalisation (lowercase alphanumeric only), so punctuation does not matter.
EXPECTED_CLASSES = {
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
}


def norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMG_EXT:
            yield path


def variant_and_class(root: Path, path: Path) -> tuple[str, str]:
    """Handle both <class>/... and raw/<variant>/<class>/... layouts."""
    parts = path.relative_to(root).parts
    for i, part in enumerate(parts[:-1]):
        if part.lower() in VARIANTS:
            return part.lower(), parts[i + 1]
    if len(parts) >= 2:
        return "default", parts[-2]
    return "default", "unknown"


def dhash(im: Image.Image, size: int = 8) -> int:
    """64-bit difference hash: robust to resizing and JPEG recompression."""
    im = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = im.tobytes()
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | int(px[base + col] > px[base + col + 1])
    return bits


def near_duplicate_pairs(hashes: list[tuple[int, str]], window: int, max_dist: int):
    """Approximate near-duplicate search: neighbouring hashes in sorted order."""
    hashes.sort()
    pairs = []
    for i, (h1, p1) in enumerate(hashes):
        for h2, p2 in hashes[i + 1 : i + 1 + window]:
            if (h1 ^ h2).bit_count() <= max_dist:
                pairs.append((p1, p2))
    return pairs


def load_splits(processed_dir: Path) -> dict[str, str]:
    """Map dataset-relative path -> split name using data/processed/*.csv manifests."""
    mapping: dict[str, str] = {}
    for name in ("train", "val", "test"):
        path = processed_dir / f"{name}.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                mapping[row["path"]] = name
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--skip-dupes", action="store_true", help="skip the duplicate scan")
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS),
                    help="limit the scan to this variant (repeatable)")
    ap.add_argument("--splits-dir", type=Path,
                    help="data/processed dir with train/val/test.csv for the leakage check")
    ap.add_argument("--window", type=int, default=48, help="neighbour window for near-dupes")
    ap.add_argument("--max-dist", type=int, default=6, help="max Hamming distance for near-dupes")
    args = ap.parse_args()

    root = args.data_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[tuple[str, str]] = Counter()
    sizes: Counter[tuple[int, int]] = Counter()
    corrupt: list[str] = []
    hashes: list[tuple[int, str]] = []
    total = 0

    for path in iter_images(root):
        variant, cls = variant_and_class(root, path)
        if args.variant and variant not in args.variant:
            continue
        total += 1
        try:
            with Image.open(path) as im:
                sizes[im.size] += 1
                digest = None if args.skip_dupes else dhash(im)
            if digest is not None:
                hashes.append((digest, str(path.relative_to(root))))
        except Exception:  # noqa: BLE001 - any decode failure counts as corrupt
            corrupt.append(str(path.relative_to(root)))
            continue
        counts[(variant, cls)] += 1

    variants = sorted({v for v, _ in counts})
    classes_by_variant = defaultdict(set)
    for (variant, cls) in counts:
        classes_by_variant[variant].add(cls)

    exact_groups: dict[int, list[str]] = defaultdict(list)
    for h, p in hashes:
        exact_groups[h].append(p)
    exact_dupes = {h: g for h, g in exact_groups.items() if len(g) > 1}
    redundant = sum(len(g) - 1 for g in exact_dupes.values())

    near_pairs = near_duplicate_pairs(hashes.copy(), args.window, args.max_dist)
    near_real = [p for p in near_pairs if p[0] != p[1]]

    split_of = load_splits(args.splits_dir) if args.splits_dir else {}

    def crosses(a: str, b: str) -> bool:
        sa, sb = split_of.get(a), split_of.get(b)
        return bool(sa and sb and sa != sb)

    near_cross = [p for p in near_real if crosses(*p)]
    exact_cross = [(g[0], p) for g in exact_dupes.values() for p in g[1:] if crosses(g[0], p)]

    # ---------- CSV ----------
    with (args.out_dir / "class_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "class", "count"])
        for (variant, cls), n in sorted(counts.items()):
            w.writerow([variant, cls, n])

    with (args.out_dir / "duplicates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "representative", "duplicate"])
        for gi, (h, group) in enumerate(sorted(exact_dupes.items(), key=lambda kv: -len(kv[1])), 1):
            for p in group[1:]:
                w.writerow([gi, group[0], p])

    with (args.out_dir / "near_duplicates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "same_dhash", "cross_split"])
        hash_of = {p: h for h, p in hashes}
        for a, b in near_real:
            w.writerow([a, b, "yes" if hash_of.get(a) == hash_of.get(b) else "no",
                        "yes" if crosses(a, b) else ""])

    # ---------- Markdown ----------
    lines: list[str] = []
    add = lines.append
    add("# Dataset audit - PlantVillage")
    add("")
    add(f"- Root: `{root}`")
    add(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    add(f"- Images scanned: **{total:,}**")
    add(f"- Corrupt / unreadable: **{len(corrupt):,}**")
    add(f"- Exact duplicate groups: **{len(exact_dupes):,}** "
        f"({redundant:,} redundant files)")
    add(f"- Near-duplicate pairs (approx., Hamming <= {args.max_dist}): **{len(near_real):,}**")
    if split_of:
        add(f"- Pairs crossing a split boundary: **{len(near_cross) + len(exact_cross):,}** "
            f"(near: {len(near_cross):,}, exact: {len(exact_cross):,})")
    add("")

    add("## Variants")
    add("")
    add("| variant | images | classes |")
    add("|---|---:|---:|")
    for v in variants:
        n = sum(c for (vv, _), c in counts.items() if vv == v)
        add(f"| {v} | {n:,} | {len(classes_by_variant[v])} |")
    add("")

    add("## Warnings")
    add("")
    warnings: list[str] = []
    for v in variants:
        if len(classes_by_variant[v]) != 38:
            warnings.append(
                f"`{v}` has {len(classes_by_variant[v])} classes, expected 38."
            )
    present = {norm(c) for cs in classes_by_variant.values() for c in cs}
    missing = sorted({norm(c) for c in EXPECTED_CLASSES} - present)
    if missing:
        warnings.append(f"{len(missing)} canonical classes not found by name "
                        f"(naming differences?): {missing[:5]}")
    if not warnings:
        warnings.append("None.")
    for w_ in warnings:
        add(f"- {w_}")
    add("")

    add("## Image dimensions")
    add("")
    add("| width x height | count |")
    add("|---|---:|")
    for (w_, h_), n in sizes.most_common(10):
        add(f"| {w_} x {h_} | {n:,} |")
    add("")

    add("## Class counts")
    add("")
    add("| class | " + " | ".join(variants) + " |")
    add("|---|" + "---:|" * len(variants))
    all_classes = sorted({c for _, c in counts})
    for cls in all_classes:
        row = [f"{counts.get((v, cls), 0):,}" for v in variants]
        add(f"| {cls} | " + " | ".join(row) + " |")
    add("")

    if corrupt:
        add("## Corrupt files (first 20)")
        add("")
        for p in corrupt[:20]:
            add(f"- `{p}`")
        add("")

    if split_of and (near_cross or exact_cross):
        add("## Cross-split duplicates (leakage)")
        add("")
        add(f"- Near-duplicate pairs spanning splits: {len(near_cross):,}")
        add(f"- Exact duplicates spanning splits: {len(exact_cross):,}")
        add("")
        for a, b in (exact_cross + near_cross)[:10]:
            add(f"- `{a}` [{split_of.get(a)}] ~ `{b}` [{split_of.get(b)}]")
        add("")

    if near_real:
        add("## Near-duplicate examples (first 20)")
        add("")
        for a, b in near_real[:20]:
            add(f"- `{a}` ~ `{b}`")
        add("")

    (args.out_dir / "dataset_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"scanned {total:,} images")
    print(f"classes per variant: " +
          ", ".join(f"{v}={len(classes_by_variant[v])}" for v in variants))
    print(f"corrupt={len(corrupt)} exact-dupe-groups={len(exact_dupes)} "
          f"near-dupe-pairs={len(near_real)}")
    if split_of:
        print(f"cross-split duplicates: near={len(near_cross)} exact={len(exact_cross)}")
    print(f"report -> {args.out_dir / 'dataset_report.md'}")


if __name__ == "__main__":
    main()
