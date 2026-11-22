#!/usr/bin/env python
"""E1 - linear probe on frozen MobileNetV3-Small features.

The cheapest meaningful baseline: cache 576-d ImageNet features once, then fit a multinomial
logistic regression on top. Establishes how separable the 38 classes are before any fine-tuning.

    python scripts/e1_linear_probe.py
    python scripts/e1_linear_probe.py --force-features   # re-extract
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ManifestImages(Dataset):
    def __init__(self, root: Path, manifest: Path, transform):
        self.root = root
        self.transform = transform
        with manifest.open(encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        with Image.open(self.root / row["path"]) as im:
            tensor = self.transform(im.convert("RGB"))
        return tensor, int(row["label_idx"])


def extract_features(args, split: str, model, transform) -> tuple[np.ndarray, np.ndarray]:
    manifest = args.processed / f"{split}.csv"
    ds = ManifestImages(args.raw, manifest, transform)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, persistent_workers=args.workers > 0,
    )
    feats, labels = [], []
    start = time.time()
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader, 1):
            feats.append(model(x).numpy())
            labels.append(y.numpy())
            if i % 50 == 0:
                done = i * args.batch_size
                rate = done / (time.time() - start)
                print(f"  [{split}] {done:,}/{len(ds):,}  {rate:.0f} img/s", flush=True)
    elapsed = time.time() - start
    X = np.concatenate(feats).astype(np.float32)
    y = np.concatenate(labels).astype(np.int64)
    print(f"[features] {split}: {X.shape} in {elapsed:.0f}s ({len(ds)/elapsed:.0f} img/s)")
    return X, y


def load_or_extract(args) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cache = args.cache
    if cache.is_file() and not args.force_features:
        print(f"[features] loading cache {cache}")
        data = np.load(cache)
        return {s: (data[f"X_{s}"], data[f"y_{s}"]) for s in ("train", "val", "test")}

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    model.classifier = torch.nn.Identity()  # 576-d pooled features
    model.eval()
    transform = weights.transforms()

    out = {}
    for split in ("train", "val", "test"):
        out[split] = extract_features(args, split, model, transform)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, **{f"X_{s}": out[s][0] for s in out}, **{f"y_{s}": out[s][1] for s in out}
    )
    print(f"[features] cached -> {cache}")
    return out


def fit_probe(X_train, y_train, X_eval, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"[probe] fit on {len(y_train):,} samples in {time.time() - t0:.0f}s")
    proba = clf.predict_proba(X_eval)
    return proba


def metrics(y_true, proba, class_names):
    from sklearn.metrics import accuracy_score, f1_score, recall_score, top_k_accuracy_score

    labels = list(range(len(class_names)))
    pred = proba.argmax(1)
    top1 = accuracy_score(y_true, pred)
    top3 = top_k_accuracy_score(y_true, proba, k=3, labels=labels)
    macro_f1 = f1_score(y_true, pred, average="macro", labels=labels, zero_division=0)
    recall = recall_score(y_true, pred, average=None, labels=labels, zero_division=0)
    return {
        "top1": float(top1),
        "top3": float(top3),
        "macro_f1": float(macro_f1),
        "recall_per_class": {class_names[i]: float(r) for i, r in enumerate(recall)},
        "pred": pred,
    }


def write_report(args, res, class_names, class_counts, secs_per_image):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    y_true = res["test"]["y"]
    cm = confusion_matrix(y_true, res["test"]["metrics"]["pred"], labels=range(len(class_names)))

    fig, ax = plt.subplots(figsize=(14, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.set(
        xticks=range(len(class_names)), yticks=range(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        ylabel="true", xlabel="predicted",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=5)
    plt.setp(ax.get_yticklabels(), fontsize=5)
    ax.set_title("E1 linear probe - test set")
    fig.tight_layout()
    fig.savefig(args.reports / "E1_confusion.png", dpi=150)
    plt.close(fig)

    rows = np.argwhere(cm > 0)
    pairs = sorted(
        ((int(cm[i, j]), class_names[i], class_names[j]) for i, j in rows if i != j),
        reverse=True,
    )[:10]

    weakest = sorted(res["test"]["metrics"]["recall_per_class"].items(), key=lambda kv: kv[1])[:8]
    sizes = res["sizes"]
    feat_line = (
        "cached (no re-extraction)"
        if secs_per_image < 1e-3
        else f"{secs_per_image * 1000:.0f} ms/image (CPU)"
    )
    lines = [
        "# E1 - linear probe on frozen MobileNetV3-Small",
        "",
        "576-d ImageNet features (224x224), multinomial logistic regression on top.",
        "",
        f"- train: {sizes['train']:,} images",
        f"- val: {sizes['val']:,} images",
        f"- test: {sizes['test']:,} images",
        f"- feature extraction: {feat_line}",
        "",
        "| fit on | evaluated on | top-1 | top-3 | macro-F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for fit_on, eval_on in (("train", "val"), ("train+val", "test")):
        m = res[eval_on]["metrics"]
        lines.append(
            f"| {fit_on} | {eval_on} | {m['top1']:.4f} | {m['top3']:.4f} | {m['macro_f1']:.4f} |"
        )
    lines += ["", "## Weakest classes (test)", "", "| class | recall | test images |", "|---|---:|---:|"]
    for name, rec in weakest:
        lines.append(f"| {name} | {rec:.3f} | {class_counts[name]:,} |")
    lines += ["", "## Top confusions (test)", "", "| count | true | predicted as |", "|---:|---|---|"]
    for n, a, b in pairs:
        lines.append(f"| {n} | {a} | {b} |")
    lines += ["", "![confusion matrix](E1_confusion.png)", ""]
    (args.reports / "E1_linear_probe.md").write_text("\n".join(lines), encoding="utf-8")

    (args.reports / "E1_metrics.json").write_text(
        json.dumps(
            {
                split: {k: v for k, v in payload["metrics"].items() if k != "pred"}
                for split, payload in res.items()
                if "metrics" in payload
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.reports / "E1_confusion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_vs_pred"] + class_names)
        for i, name in enumerate(class_names):
            w.writerow([name] + cm[i].tolist())
    print(f"[report] {args.reports / 'E1_linear_probe.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/plantvillage_hf"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--cache", type=Path, default=Path("data/features/mobilenetv3_small.npz"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-features", action="store_true")
    args = ap.parse_args()

    args.reports.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(12)
    started = time.time()

    data = load_or_extract(args)
    extract_secs = time.time() - started
    n_images = sum(len(v[1]) for v in data.values())
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    index = json.loads((args.processed / "class_index.json").read_text(encoding="utf-8"))
    class_names = [name for name, _ in sorted(index.items(), key=lambda kv: kv[1])]

    res: dict[str, dict] = {}
    # protocol: fit on train, evaluate val (model selection happens here)
    proba_val = fit_probe(X_train, y_train, X_val, args.seed)
    res["val"] = {"y": y_val, "metrics": metrics(y_val, proba_val, class_names)}
    print(f"[val] top1={res['val']['metrics']['top1']:.4f} "
          f"top3={res['val']['metrics']['top3']:.4f} "
          f"macroF1={res['val']['metrics']['macro_f1']:.4f}")

    # final: refit on train+val, report on the untouched test set
    X_fit = np.concatenate([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    proba_test = fit_probe(X_fit, y_fit, X_test, args.seed)
    res["test"] = {"y": y_test, "metrics": metrics(y_test, proba_test, class_names)}
    print(f"[test] top1={res['test']['metrics']['top1']:.4f} "
          f"top3={res['test']['metrics']['top3']:.4f} "
          f"macroF1={res['test']['metrics']['macro_f1']:.4f}")
    res["sizes"] = {"train": len(y_train), "val": len(y_val), "test": len(y_test)}

    counts = Counter(y_test.tolist())
    class_counts = {class_names[i]: counts.get(i, 0) for i in range(len(class_names))}
    per_image = extract_secs / max(1, n_images)
    write_report(args, res, class_names, class_counts, per_image)


if __name__ == "__main__":
    main()
