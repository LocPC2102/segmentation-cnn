"""Evaluate alexnet_unet_tuned_best.pt on train/val/test and compare it with the earlier models.

Reuses run_split / plot_confusions from report_alexnet.py. Nothing is overwritten:
results go to src/cnn_segmentation/eval/alexnet_tuned/.

Run from the project root:
    python -m src.cnn_segmentation.eval.report_alexnet_tuned

Comparison columns appear for every baseline metrics file that exists:
    LeNet   : eval/all_splits_metrics.json
    AlexNet : eval/alexnet/metrics.json
"""

import argparse
import json
from pathlib import Path

import torch

from src.cnn_segmentation.datasets.transforms import CLASS_NAMES, NUM_CLASSES
from src.cnn_segmentation.eval.report_alexnet import SPLITS, plot_confusions, run_split
from src.cnn_segmentation.eval.report_metrics import load_checkpoint
from src.cnn_segmentation.models.alexnet_unet import AlexNetUNet

BASELINES = {
    "LeNet": "src/cnn_segmentation/eval/all_splits_metrics.json",
    "AlexNet": "src/cnn_segmentation/eval/alexnet/metrics.json",
}


def load_baselines():
    found = {}
    for name, path in BASELINES.items():
        if Path(path).is_file():
            found[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    return found


def write_report(results, n_params, ckpt, path, baselines):
    lines = ["# AlexNet U-Net (tuned) segmentation report", "",
             f"- Checkpoint: `{ckpt}`", f"- Parameters: {n_params:,}", "",
             "## Overall", "",
             "| Split | Images | Pixel Acc | mIoU | mDice |", "|---|---|---|---|---|"]
    for s in SPLITS:
        r = results[s]
        lines.append(f"| {s} | {r['num_images']} | {r['pixel_accuracy']:.4f} | {r['mean_iou']:.4f} | {r['mean_dice']:.4f} |")

    for s in ("val", "test"):
        lines += ["", f"## Per-class ({s})", "", "| Class | IoU | Dice | Recall |", "|---|---|---|---|"]
        for name, m in results[s]["per_class"].items():
            lines.append(f"| {name} | {m['iou']:.4f} | {m['dice']:.4f} | {m['pixel_accuracy_recall']:.4f} |")

    models = {**baselines, "AlexNet tuned": results}
    lines += ["", "## Comparison (test)", "",
              "| Metric | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    rows = [(label, lambda r, k=key: r["test"][k])
            for key, label in (("pixel_accuracy", "Pixel Acc"), ("mean_iou", "mIoU"), ("mean_dice", "mDice"))]
    for c in CLASS_NAMES:
        rows.append((f"IoU {c}", lambda r, c=c: r["test"]["per_class"][c]["iou"]))
        rows.append((f"Dice {c}", lambda r, c=c: r["test"]["per_class"][c]["dice"]))
    for label, get in rows:
        lines.append(f"| {label} | " + " | ".join(f"{get(r):.4f}" for r in models.values()) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="src/cnn_segmentation/models/checkpoints_4cls/alexnet_unet_tuned_best.pt")
    ap.add_argument("--out-dir", default="src/cnn_segmentation/eval/alexnet_tuned")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNetUNet(NUM_CLASSES).to(device)
    load_checkpoint(model, Path(args.checkpoint), device)
    n_params = sum(p.numel() for p in model.parameters())

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split in SPLITS:
        results[split] = r = run_split(model, split, args.size, args.batch_size, device)
        print(f"{split:5s} | acc {r['pixel_accuracy']:.4f} | mIoU {r['mean_iou']:.4f} | mDice {r['mean_dice']:.4f}")

    print("\nTEST per class")
    for name, m in results["test"]["per_class"].items():
        print(f"  {name:<10s}: IoU {m['iou']:.4f} | Dice {m['dice']:.4f} | recall {m['pixel_accuracy_recall']:.4f}")

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_confusions(results, out_dir / "confusion_matrices_all_splits.png", out_dir / "confusion_matrix_test.png")
    write_report(results, n_params, args.checkpoint, out_dir / "report.md", load_baselines())
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()