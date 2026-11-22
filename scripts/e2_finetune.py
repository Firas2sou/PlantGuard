#!/usr/bin/env python
"""Fine-tune a backbone for 38-class leaf classification (experiments E2/E3/E4).

    python scripts/e2_finetune.py                        # E2: MobileNetV3-Small, basic aug
    python scripts/e2_finetune.py --strong-aug           # E3: + color/rotation/sharpness
    python scripts/e2_finetune.py --bkg-randomize 0.3    # E4: + random backgrounds (seg alpha)
    python scripts/e2_finetune.py --backbone efficientnet_b0 --epochs 12

Works on ROCm (AMD) / CUDA if available, otherwise CPU. Logs every epoch to
reports/training_log.csv and writes a per-experiment card plus an ONNX export.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

if torch.cuda.is_available():
    # MIOpen JIT compilation is broken in the pip-installed ROCm SDK on Windows;
    # PyTorch's native HIP kernels are a correct (slightly slower) fallback.
    torch.backends.cudnn.enabled = False


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LeafDataset(Dataset):
    def __init__(self, root: Path, manifest: Path, transform, bkg_prob: float = 0.0):
        self.root = root
        self.transform = transform
        self.bkg_prob = bkg_prob
        with manifest.open(encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        with Image.open(self.root / row["path"]) as im:
            img = im.convert("RGB")
        if self.bkg_prob > 0 and random.random() < self.bkg_prob:
            img = self._randomize_background(img, row["path"])
        return self.transform(img), int(row["label_idx"])

    def _randomize_background(self, img: Image.Image, path: str) -> Image.Image:
        """Paste the leaf (alpha from the segmented twin, black = background) onto a
        procedural random background. Falls back to the original image when the
        segmented twin is missing."""
        p = Path(path)
        seg_dir = Path(str(p.parent).replace("/color/", "/segmented/")
                       .replace("\\color\\", "\\segmented\\"))
        seg_file = self.root / seg_dir / (p.stem + "_final_masked.jpg")
        if not seg_file.is_file():
            return img
        with Image.open(seg_file) as seg:
            seg = seg.convert("RGB")
        seg_arr = np.asarray(seg).astype(np.float32)
        alpha = (seg_arr.sum(axis=2) > 24).astype(np.float32)
        if alpha.mean() < 0.02:
            return img
        leaf = np.asarray(img).astype(np.float32)
        out = leaf * alpha[..., None] + self._make_background(img.size) * (1 - alpha[..., None])
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    @staticmethod
    def _make_background(size: tuple[int, int]) -> np.ndarray:
        rng = np.random.default_rng()
        h, w = size[1], size[0]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        base = rng.uniform(0, 255, size=3).astype(np.float32)
        gradient = (
            x / w * rng.uniform(-90, 90)
            + y / h * rng.uniform(-90, 90)
        )
        bg = base[None, None, :] + gradient[..., None]
        noise = rng.normal(0, 14, size=(h, w, 3)).astype(np.float32)
        return np.clip(bg + noise, 0, 255)


def make_transforms(train: bool, strong: bool):
    if train:
        ops = [T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip()]
        if strong:
            ops += [
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.RandomRotation(20),
                T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.2),
            ]
        ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    else:
        ops = [T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(ops)


def build_model(name: str, n_classes: int) -> nn.Module:
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    else:
        raise SystemExit(f"unknown backbone: {name}")
    return m


def np_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = proba.argmax(1)
    n = proba.shape[1]
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, pred), 1)
    tp = np.diagonal(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    top3 = np.mean(np.any(np.argsort(-proba, axis=1)[:, :3] == y_true[:, None], axis=1))
    return {
        "top1": float((pred == y_true).mean()),
        "top3": float(top3),
        "macro_f1": float(f1.mean()),
        "pred": pred,
    }


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probas, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            out = model(x.to(device))
            probas.append(torch.softmax(out, 1).float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probas), np.concatenate(labels)


def class_weights(manifest: Path, index: dict[str, int]) -> torch.Tensor:
    counts: Counter[str] = Counter()
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            counts[row["label"]] += 1
    total = sum(counts.values())
    w = np.ones(len(index), dtype=np.float32)
    for label, c in counts.items():
        w[index[label]] = np.sqrt(total / c)  # sqrt-inverse: moderate
    return torch.from_numpy(w / w.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/plantvillage_hf"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--backbone", default="mobilenet_v3_small")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--strong-aug", action="store_true")
    ap.add_argument("--bkg-randomize", type=float, default=0.0,
                    help="probability of compositing a random background (E4)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs-per-eval", type=int, default=1)
    ap.add_argument("--no-class-weights", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    index = json.loads((args.processed / "class_index.json").read_text(encoding="utf-8"))
    class_names = [name for name, _ in sorted(index.items(), key=lambda kv: kv[1])]
    n_classes = len(class_names)

    train_ds = LeafDataset(
        args.raw, args.processed / "train.csv",
        make_transforms(True, args.strong_aug), bkg_prob=args.bkg_randomize,
    )
    eval_tf = make_transforms(False, False)
    val_ds = LeafDataset(args.raw, args.processed / "val.csv", eval_tf)
    test_ds = LeafDataset(args.raw, args.processed / "test.csv", eval_tf)

    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=(device == "cuda"), persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

    model = build_model(args.backbone, n_classes).to(device)
    weights = None if args.no_class_weights else class_weights(args.processed / "train.csv", index).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    tag = args.backbone + ("_strong" if args.strong_aug else "") + (
        f"_bkg{args.bkg_randomize:.0%}" if args.bkg_randomize > 0 else "")
    exp_id = f"E2_{tag}"
    log_path = args.reports / "training_log.csv"
    log_exists = log_path.is_file()

    best_val, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(x)
                    loss = criterion(out, y)
            else:
                out = model(x)
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        scheduler.step()
        secs = time.time() - t0

        if epoch % args.epochs_per_eval == 0 or epoch == args.epochs:
            proba, y_true = evaluate(model, val_loader, device)
            m = np_metrics(y_true, proba)
            print(f"[epoch {epoch:02d}/{args.epochs}] loss={total_loss/len(train_ds):.4f} "
                  f"{secs:.0f}s/epoch | val top1={m['top1']:.4f} macroF1={m['macro_f1']:.4f}",
                  flush=True)
            if m["top1"] > best_val:
                best_val = m["top1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            with log_path.open("a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if not log_exists:
                    w.writerow(["exp", "epoch", "loss", "val_top1", "val_top3", "val_macro_f1",
                                "sec_per_epoch", "device", "lr"])
                    log_exists = True
                w.writerow([exp_id, epoch, f"{total_loss/len(train_ds):.4f}",
                            f"{m['top1']:.4f}", f"{m['top3']:.4f}", f"{m['macro_f1']:.4f}",
                            f"{secs:.0f}", device, args.lr])
        else:
            print(f"[epoch {epoch:02d}/{args.epochs}] {secs:.0f}s/epoch", flush=True)

    model.load_state_dict(best_state)
    ckpt = args.models / f"{exp_id}.pt"
    args.models.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "args": vars(args), "class_names": class_names}, ckpt)
    print(f"[save] {ckpt} (best val top1={best_val:.4f})")

    t0 = time.time()
    proba, y_true = evaluate(model, test_loader, device)
    test_secs = time.time() - t0
    m = np_metrics(y_true, proba)
    print(f"[test] top1={m['top1']:.4f} top3={m['top3']:.4f} macroF1={m['macro_f1']:.4f} "
          f"({len(y_true)/test_secs:.0f} img/s)")

    # ONNX export (CPU, same weights)
    try:
        onnx_path = args.models / f"{exp_id}.onnx"
        model.eval().cpu()
        torch.onnx.export(
            model, torch.randn(1, 3, 224, 224), str(onnx_path),
            opset_version=17, input_names=["image"], output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}},
        )
        print(f"[onnx] {onnx_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[onnx] export failed: {exc}")

    write_card(args, exp_id, class_names, y_true, proba, m, ckpt, test_secs)


def write_card(args, exp_id, class_names, y_true, proba, m, ckpt, test_secs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    np.add.at(cm, (y_true, m["pred"]), 1)

    fig, ax = plt.subplots(figsize=(14, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names, ylabel="true", xlabel="predicted")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=5)
    plt.setp(ax.get_yticklabels(), fontsize=5)
    ax.set_title(f"{exp_id} - test set")
    fig.tight_layout()
    fig.savefig(args.reports / f"{exp_id}_confusion.png", dpi=150)
    plt.close(fig)

    tp = np.diagonal(cm).astype(np.float64)
    rec = tp / (cm.sum(1) + 1e-9)
    weakest = sorted(zip(class_names, rec.tolist()), key=lambda kv: kv[1])[:8]
    pairs = sorted(
        ((int(cm[i, j]), class_names[i], class_names[j])
         for i, j in np.argwhere(cm > 0) if i != j),
        reverse=True,
    )[:10]

    n_imgs = len(y_true)
    lines = [
        f"# {exp_id} - fine-tuned {args.backbone}",
        "",
        f"- device: {'GPU' if args.device != 'cpu' else 'CPU'}, batch {args.batch_size}, "
        f"{args.epochs} epochs, lr {args.lr}, label smoothing {args.label_smoothing}",
        f"- strong augmentation: {args.strong_aug}, random background prob: {args.bkg_randomize}",
        f"- class-weighted loss: {not args.no_class_weights}",
        f"- test: {n_imgs:,} images at {n_imgs/test_secs:.0f} img/s",
        "",
        f"| metric | value |",
        f"|---|---:|",
        f"| top-1 | {m['top1']:.4f} |",
        f"| top-3 | {m['top3']:.4f} |",
        f"| macro-F1 | {m['macro_f1']:.4f} |",
        "",
        "## Weakest classes (test)",
        "",
        "| class | recall |",
        "|---|---:|",
    ]
    for name, r in weakest:
        lines.append(f"| {name} | {r:.3f} |")
    lines += ["", "## Top confusions (test)", "", "| count | true | predicted as |", "|---:|---|---|"]
    for n, a, b in pairs:
        lines.append(f"| {n} | {a} | {b} |")
    lines += ["", f"![confusion matrix]({exp_id}_confusion.png)", ""]
    (args.reports / f"{exp_id}.md").write_text("\n".join(lines), encoding="utf-8")

    with (args.reports / f"{exp_id}_confusion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_vs_pred"] + class_names)
        for i, name in enumerate(class_names):
            w.writerow([name] + cm[i].tolist())
    print(f"[report] {args.reports / (exp_id + '.md')}")


if __name__ == "__main__":
    main()
