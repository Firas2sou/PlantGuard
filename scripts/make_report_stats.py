#!/usr/bin/env python
"""Compute dataset statistics for the report and render the class-distribution chart.

    python scripts/make_report_stats.py --lang fr
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\plant-doctor")
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "plantvillage_hf" / "raw"
REP = ROOT / "reports"


def load(name: str) -> list[dict]:
    with (PROC / f"{name}.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=["en", "fr"], default="en")
    args = ap.parse_args()

    splits = {s: load(s) for s in ("train", "val", "test")}
    index = json.loads((PROC / "class_index.json").read_text(encoding="utf-8"))
    classes = [n for n, _ in sorted(index.items(), key=lambda kv: kv[1])]

    variants = {}
    for v in ("color", "grayscale", "segmented"):
        vdir = RAW / v
        n = sum(
            1 for p in vdir.rglob("*")
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        n_classes = sum(1 for d in vdir.iterdir() if d.is_dir())
        variants[v] = {"images": n, "classes": n_classes}

    counts = {s: Counter(r["label"] for r in rows) for s, rows in splits.items()}
    total = {c: sum(counts[s][c] for s in ("train", "val", "test")) for c in classes}

    crop_info: dict[str, dict] = defaultdict(lambda: {"classes": set(), "train": 0, "val": 0, "test": 0})
    for s, rows in splits.items():
        for r in rows:
            crop = r["label"].partition("___")[0]
            crop_info[crop]["classes"].add(r["label"])
            crop_info[crop][s] += 1

    split_of = {r["path"]: s for s, rows in splits.items() for r in rows}
    exact = list(csv.DictReader((REP / "duplicates.csv").open(encoding="utf-8", newline="")))
    near = list(csv.DictReader((REP / "near_duplicates.csv").open(encoding="utf-8", newline="")))
    cross_exact = sum(
        1 for row in exact
        if split_of.get(row["representative"]) and split_of.get(row["duplicate"])
        and split_of[row["representative"]] != split_of[row["duplicate"]]
    )
    cross_near = sum(1 for row in near if row["cross_split"] == "yes")

    stats = {
        "variants": variants,
        "split_sizes": {s: len(rows) for s, rows in splits.items()},
        "n_classes": len(classes),
        "total_images": sum(total.values()),
        "balance": {
            "train_min": (min(counts["train"], key=counts["train"].get), min(counts["train"].values())),
            "train_max": (max(counts["train"], key=counts["train"].get), max(counts["train"].values())),
            "test_min": (min(counts["test"], key=counts["test"].get), min(counts["test"].values())),
            "test_max": (max(counts["test"], key=counts["test"].get), max(counts["test"].values())),
        },
        "crops": [
            {
                "crop": crop,
                "classes": len(info["classes"]),
                "train": info["train"],
                "val": info["val"],
                "test": info["test"],
            }
            for crop, info in sorted(crop_info.items())
        ],
        "duplicates": {
            "exact_groups": len(exact),
            "near_pairs": len(near),
            "cross_split_exact": cross_exact,
            "cross_split_near": cross_near,
        },
        "classes": {c: {"train": counts["train"][c], "val": counts["val"][c],
                        "test": counts["test"][c]} for c in sorted(classes)},
    }
    (REP / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ---- class distribution chart (stacked horizontal bars) ----
    if args.lang == "fr":
        legend_train, legend_test = "entraînement + validation", "test"
        chart_title = "Distribution des classes — variante couleur de PlantVillage (38 classes)"
        xlabel, chart_out = "images", REP / "class_distribution_fr.png"
    else:
        legend_train, legend_test = "train + validation", "test"
        chart_title = "Class distribution - PlantVillage color variant (38 classes)"
        xlabel, chart_out = "images", REP / "class_distribution.png"

    order = sorted(classes, key=lambda c: -total[c])
    ypos = list(range(len(order)))
    tr = [counts["train"][c] + counts["val"][c] for c in order]
    te = [counts["test"][c] for c in order]
    fig, ax = plt.subplots(figsize=(7.5, 9.5))
    ax.barh(ypos, tr, color="#4C72B0", label=legend_train)
    ax.barh(ypos, te, left=tr, color="#DD8452", label=legend_test)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order, fontsize=6)
    ax.set_xlabel(xlabel)
    ax.set_title(chart_title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(chart_out, dpi=150)
    plt.close(fig)

    print(json.dumps(stats["split_sizes"]), "| crops:", len(stats["crops"]))
    print("balance:", stats["balance"])
    print("dupes:", stats["duplicates"])
    print("chart ->", chart_out)


if __name__ == "__main__":
    main()
