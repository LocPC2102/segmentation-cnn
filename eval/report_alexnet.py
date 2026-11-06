"""Evaluate alexnet_unet_best.pt on train/val/test and export metrics, confusion matrices and a report.

Run from the project root:
    python -m src.cnn_segmentation.eval.report_alexnet

Outputs go to src/cnn_segmentation/eval/alexnet/:
    metrics.json, confusion_matrices_all_splits.png, confusion_matrix_test.png, report.md

If eval/all_splits_metrics.json (LeNet) exists, the report also compares the two models.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cnn_segmentation.datasets.dataset import AxialSegmentationDataset
from src.cnn_segmentation.datasets.transforms import (
    CLASS_NAMES, NUM_CLASSES, DualCompose, DualResize, DualToTensor)
from src.cnn_segmentation.eval.report_metrics import evaluate, load_checkpoint
from src.cnn_segmentation.models.alexnet_unet import AlexNetUNet

SPLITS = ("train", "val", "test")


def run_split(model, split, size, batch_size, device):
    transform = DualCompose([DualResize(size), DualToTensor()])
    dataset = AxialSegmentationDataset(f"DATA/processed/seg_{split}.csv", transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
    return evaluate(model, loader, device, NUM_CLASSES)


def fmt_count(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    return f"{n / 1_000:.1f}k" if n >= 1_000 else str(n)


def draw_cm(ax, res, title):
    """Row-normalised confusion matrix (recall %) annotated with pixel counts."""
    cm = np.array(res["confusion_matrix"], dtype=np.float64)
    rows = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, rows, out=np.zeros_like(cm), where=rows != 0) * 100

    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{pct[i, j]:.1f}%\n({fmt_count(int(cm[i, j]))})", ha="center", va="center",
                    color="white" if pct[i, j] > 50 else "black", fontsize=9)
    return im


def plot_confusions(results, all_path, test_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=200)
    for ax, split in zip(axes, SPLITS):
        im = draw_cm(ax, results[split], f"{split.upper()} (Acc {results[split]['pixel_accuracy'] * 100:.2f}%)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Recall (%)")
    fig.savefig(all_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    t = results["test"]
    im = draw_cm(ax, t, f"AlexNet U-Net, TEST\nAcc {t['pixel_accuracy'] * 100:.2f}% | mIoU {t['mean_iou']:.4f}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Recall (%)")
    fig.tight_layout()
    fig.savefig(test_path, bbox_inches="tight")
    plt.close(fig)


def write_report(results, n_params, ckpt, path, lenet=None):
    lines = ["# AlexNet U-Net segmentation report", "",
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

    if lenet:
        lines += ["", "## AlexNet vs LeNet (test)", "",
                  "| Metric | LeNet | AlexNet |", "|---|---|---|"]
        a, l = results["test"], lenet["test"]
        for key, label in (("pixel_accuracy", "Pixel Acc"), ("mean_iou", "mIoU"), ("mean_dice", "mDice")):
            lines.append(f"| {label} | {l[key]:.4f} | {a[key]:.4f} |")
        for name in a["per_class"]:
            lines.append(f"| IoU {name} | {l['per_class'][name]['iou']:.4f} | {a['per_class'][name]['iou']:.4f} |")
            lines.append(f"| Dice {name} | {l['per_class'][name]['dice']:.4f} | {a['per_class'][name]['dice']:.4f} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="src/cnn_segmentation/models/checkpoints_4cls/alexnet_unet_best.pt")
    ap.add_argument("--lenet-metrics", default="src/cnn_segmentation/eval/all_splits_metrics.json")
    ap.add_argument("--out-dir", default="src/cnn_segmentation/eval/alexnet")
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
        results[split] = run_split(model, split, args.size, args.batch_size, device)
        r = results[split]
        print(f"{split:5s} | acc {r['pixel_accuracy']:.4f} | mIoU {r['mean_iou']:.4f} | mDice {r['mean_dice']:.4f}")

    print("\nTEST per class")
    for name, m in results["test"]["per_class"].items():
        print(f"  {name:<10s}: IoU {m['iou']:.4f} | Dice {m['dice']:.4f} | recall {m['pixel_accuracy_recall']:.4f}")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    plot_confusions(results, out_dir / "confusion_matrices_all_splits.png", out_dir / "confusion_matrix_test.png")

    lenet_path = Path(args.lenet_metrics)
    lenet = json.loads(lenet_path.read_text(encoding="utf-8")) if lenet_path.is_file() else None
    write_report(results, n_params, args.checkpoint, out_dir / "report.md", lenet)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()