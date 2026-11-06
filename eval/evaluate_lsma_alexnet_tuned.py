"""Evaluate alexnet_unet_tuned_best.pt on the external LSMA dataset and compare with earlier models.

Reuses the dataset and metric code of evaluate_lsma.py. Nothing is overwritten:
the result goes to eval/lsma_alexnet_tuned_metrics.json.

Run from the project root:
    python -m src.cnn_segmentation.eval.evaluate_lsma_alexnet_tuned

Comparison columns appear for every baseline file that exists:
    LeNet   : eval/lsma_metrics.json
    AlexNet : eval/lsma_alexnet_metrics.json
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.cnn_segmentation.datasets.transforms import CLASS_NAMES
from src.cnn_segmentation.eval.evaluate_lsma import LSMASegmentationDataset, evaluate
from src.cnn_segmentation.eval.report_metrics import load_checkpoint
from src.cnn_segmentation.models.alexnet_unet import AlexNetUNet

BASELINES = {
    "LeNet": "src/cnn_segmentation/eval/lsma_metrics.json",
    "AlexNet": "src/cnn_segmentation/eval/lsma_alexnet_metrics.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="DATA/raw/LSMA_PQR/images_png")
    ap.add_argument("--labels-dir", default="DATA/raw/LSMA_PQR/labels_yolo")
    ap.add_argument("--checkpoint", default="src/cnn_segmentation/models/checkpoints_4cls/alexnet_unet_tuned_best.pt")
    ap.add_argument("--output", default="src/cnn_segmentation/eval/lsma_alexnet_tuned_metrics.json")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LSMASegmentationDataset(Path(args.images_dir), Path(args.labels_dir), args.size)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.workers)

    model = AlexNetUNet(len(CLASS_NAMES)).to(device)
    load_checkpoint(model, Path(args.checkpoint), device)

    results = evaluate(model, loader, device, len(CLASS_NAMES))
    results.update(model="AlexNetUNet (tuned)", checkpoint=args.checkpoint,
                   input_size=[args.size, args.size], class_names=list(CLASS_NAMES))

    print(f"Images            : {results['num_images']}")
    print(f"FG macro precision: {results['foreground_macro_precision']:.4f}")
    print(f"FG macro F1       : {results['foreground_macro_f1']:.4f}")
    print(f"Pixel accuracy    : {results['pixel_accuracy']:.4f}")
    print(f"\n{'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for name in CLASS_NAMES:
        m = results["per_class"][name]
        print(f"{name:<14} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    models = {name: json.loads(Path(p).read_text(encoding="utf-8"))
              for name, p in BASELINES.items() if Path(p).is_file()}
    models["AlexNet tuned"] = results
    print(f"\n{'F1':<14}" + "".join(f"{name:>15}" for name in models))
    for name in CLASS_NAMES:
        print(f"{name:<14}" + "".join(f"{r['per_class'][name]['f1']:>15.4f}" for r in models.values()))
    print(f"{'FG macro':<14}" + "".join(f"{r['foreground_macro_f1']:>15.4f}" for r in models.values()))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nMetrics saved: {output}")


if __name__ == "__main__":
    main()