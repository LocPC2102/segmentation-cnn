"""Evaluate the segmentation checkpoint on LSMA YOLO polygon annotations.

Run from the project root:
    python -m src.cnn_segmentation.eval.evaluate_lsma

LSMA polygon classes map to model classes as follows:
    YOLO 0 -> IVD (model 3), YOLO 1 -> PE (model 1), YOLO 2 -> CC (model 2)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from src.cnn_segmentation.datasets.transforms import CLASS_NAMES
from src.cnn_segmentation.eval.report_metrics import load_checkpoint
from src.cnn_segmentation.models.lenet5_unet import LeNetUNet

YOLO_TO_MODEL = {0: 3, 1: 1, 2: 2}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


class LSMASegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, size: int):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.size = size
        self.image_paths = sorted(
            path for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {images_dir}")

        missing_labels = [
            path for path in self.image_paths
            if not (labels_dir / f"{path.stem}.txt").is_file()
        ]
        if missing_labels:
            examples = ", ".join(path.name for path in missing_labels[:5])
            raise FileNotFoundError(
                f"{len(missing_labels)} images have no matching YOLO label in "
                f"{labels_dir}; examples: {examples}"
            )

    def __len__(self):
        return len(self.image_paths)

    def _read_target(self, label_path: Path, size: tuple[int, int]) -> tuple[Image.Image, int]:
        width, height = size
        polygons = {class_id: [] for class_id in YOLO_TO_MODEL}
        with label_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                fields = line.split()
                if not fields:
                    continue
                try:
                    yolo_id = int(fields[0])
                    coords = [float(value) for value in fields[1:]]
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid annotation at {label_path}:{line_number}"
                    ) from exc

                if yolo_id not in YOLO_TO_MODEL:
                    raise ValueError(
                        f"Unsupported YOLO class {yolo_id} at {label_path}:{line_number}"
                    )
                if len(coords) < 6 or len(coords) % 2:
                    raise ValueError(
                        f"Expected at least 3 polygon points at {label_path}:{line_number}"
                    )
                if any(value < 0 or value > 1 for value in coords):
                    raise ValueError(
                        f"Polygon coordinates must be normalized to [0, 1] at "
                        f"{label_path}:{line_number}"
                    )

                points = [
                    (round(coords[i] * (width - 1)), round(coords[i + 1] * (height - 1)))
                    for i in range(0, len(coords), 2)
                ]
                polygons[yolo_id].append(points)

        target = Image.new("L", size, 0)
        draw = ImageDraw.Draw(target)
        # If polygons overlap, the more specific structures (PE, then CC) take
        # precedence over IVD, matching the model's mutually exclusive labels.
        for yolo_id in (0, 1, 2):
            for points in polygons[yolo_id]:
                draw.polygon(points, fill=YOLO_TO_MODEL[yolo_id])

        overlap_pixels = 0
        class_masks = []
        for yolo_id in YOLO_TO_MODEL:
            class_mask = Image.new("L", size, 0)
            class_draw = ImageDraw.Draw(class_mask)
            for points in polygons[yolo_id]:
                class_draw.polygon(points, fill=1)
            class_masks.append(np.asarray(class_mask, dtype=bool))
        if class_masks:
            overlap_pixels = int(np.sum(np.sum(class_masks, axis=0) > 1))

        return target, overlap_pixels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as source:
            image = source.convert("L")
        target, overlap_pixels = self._read_target(label_path, image.size)

        resize = (self.size, self.size)
        image = image.resize(resize, Image.Resampling.BILINEAR)
        target = target.resize(resize, Image.Resampling.NEAREST)

        x = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0)
        x = (x - x.mean()) / (x.std() + 1e-6)
        y = torch.from_numpy(np.array(target, dtype=np.uint8).astype(np.int64))
        return x, y, overlap_pixels


@torch.inference_mode()
def evaluate(model, loader, device, num_classes):
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    num_images = 0
    overlap_pixels = 0

    for images, masks, overlaps in loader:
        predictions = model(images.to(device)).argmax(dim=1)
        masks = masks.to(device)
        encoded = masks.flatten() * num_classes + predictions.flatten()
        confusion += torch.bincount(
            encoded, minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)
        num_images += images.shape[0]
        overlap_pixels += int(overlaps.sum().item())

    cm = confusion.to(torch.float64)
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    def ratio(numerator, denominator):
        return numerator / denominator if denominator > 0 else None

    per_class = {}
    for class_id, name in enumerate(CLASS_NAMES):
        precision = ratio(tp[class_id].item(), (tp[class_id] + fp[class_id]).item())
        recall = ratio(tp[class_id].item(), (tp[class_id] + fn[class_id]).item())
        f1 = ratio(
            (2 * tp[class_id]).item(),
            (2 * tp[class_id] + fp[class_id] + fn[class_id]).item(),
        )
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(tp[class_id].item()),
            "fp": int(fp[class_id].item()),
            "fn": int(fn[class_id].item()),
        }

    foreground = [per_class[name] for name in CLASS_NAMES[1:]]
    macro_precision = float(np.mean([item["precision"] for item in foreground]))
    macro_f1 = float(np.mean([item["f1"] for item in foreground]))
    total = cm.sum().item()
    return {
        "num_images": num_images,
        "num_pixels_evaluated": int(total),
        "pixel_accuracy": ratio(tp.sum().item(), total),
        "foreground_macro_precision": macro_precision,
        "foreground_macro_f1": macro_f1,
        "per_class": per_class,
        "overlapping_annotation_pixels_before_resize": overlap_pixels,
        "confusion_matrix": confusion.cpu().tolist(),
        "confusion_matrix_note": "Rows=true class, columns=predicted class",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LeNetUNet on LSMA segmentation polygon labels"
    )
    parser.add_argument("--images-dir", default="DATA/raw/LSMA_PQR/images_png")
    parser.add_argument("--labels-dir", default="DATA/raw/LSMA_PQR/labels_yolo")
    parser.add_argument(
        "--checkpoint",
        default="src/cnn_segmentation/models/checkpoints_4cls/lenet_unet_best.pt",
    )
    parser.add_argument(
        "--output", default="src/cnn_segmentation/eval/lsma_metrics.json"
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    project_root = Path.cwd()
    images_dir = project_root / args.images_dir
    labels_dir = project_root / args.labels_dir
    checkpoint_path = project_root / args.checkpoint
    output_path = project_root / args.output
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = LSMASegmentationDataset(images_dir, labels_dir, args.size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNetUNet(num_classes=len(CLASS_NAMES)).to(device)
    load_checkpoint(model, checkpoint_path, device)

    results = evaluate(model, loader, device, len(CLASS_NAMES))
    results.update({
        "model": "LeNetUNet",
        "checkpoint": str(checkpoint_path),
        "input_size": [args.size, args.size],
        "device": str(device),
        "class_names": list(CLASS_NAMES),
        "lsma_to_model_class_mapping": {
            "0": "IVD",
            "1": "PE",
            "2": "CC",
        },
        "overlap_policy": "CC overrides PE, PE overrides IVD",
    })

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Images     : {results['num_images']}")
    print(f"Input size : {args.size} x {args.size}")
    print(f"Device     : {device}")
    print(f"FG macro precision: {results['foreground_macro_precision']:.4f}")
    print(f"FG macro F1       : {results['foreground_macro_f1']:.4f}")
    print(f"Pixel accuracy    : {results['pixel_accuracy']:.4f}")
    print("\nPER-CLASS METRICS")
    print(f"{'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for name in CLASS_NAMES:
        metrics = results["per_class"][name]
        print(
            f"{name:<14} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nMetrics saved: {output_path}")


if __name__ == "__main__":
    main()
