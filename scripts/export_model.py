#!/usr/bin/env python
"""Export a trained checkpoint to ONNX (legacy exporter, no onnxscript needed).

    python scripts/export_model.py models/E2_mobilenet_v3_small.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from e2_finetune import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    n_classes = len(ckpt["class_names"])
    model = build_model(ckpt["args"]["backbone"], n_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out = args.out or args.checkpoint.with_suffix(".onnx")
    try:
        torch.onnx.export(
            model,
            torch.randn(1, 3, 224, 224),
            str(out),
            opset_version=args.opset,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}},
            dynamo=False,
        )
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"export failed: {exc}")
    classes_out = out.with_suffix(".classes.json")
    classes_out.write_text(
        json.dumps(ckpt["class_names"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"exported -> {out}")
    print(f"class names -> {classes_out}")


if __name__ == "__main__":
    main()
