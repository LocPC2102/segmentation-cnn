"""Train only the AlexNet-UNet segmentation model on AISSLab Axial.

Same data, loss, metrics, seed and defaults as train.py, so the result is
directly comparable with the existing LeNet-UNet checkpoint.

Run from the project root:
    python -m src.cnn_segmentation.train_alexnet --epochs 100
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cnn_segmentation.datasets.dataset import AxialSegmentationDataset
from src.cnn_segmentation.datasets.transforms import (
    CLASS_NAMES, NUM_CLASSES, DualCompose, DualRandomHFlip, DualRandomRotation, DualResize, DualToTensor)
from src.cnn_segmentation.eval.metrics import evaluate
from src.cnn_segmentation.losses.losses import DiceCELoss
from src.cnn_segmentation.models.alexnet_unet import AlexNetUNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="src/cnn_segmentation/models/checkpoints_4cls")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = DualCompose([DualResize(args.size), DualRandomHFlip(), DualRandomRotation(10), DualToTensor()])
    eval_tf = DualCompose([DualResize(args.size), DualToTensor()])

    def loader(split, tf, shuffle):
        ds = AxialSegmentationDataset(f"DATA/processed/seg_{split}.csv", transform=tf)
        return DataLoader(ds, args.batch_size, shuffle=shuffle, num_workers=args.workers)

    train_dl, val_dl, test_dl = loader("train", train_tf, True), loader("val", eval_tf, False), loader("test", eval_tf, False)

    model = AlexNetUNet(NUM_CLASSES).to(device)
    criterion = DiceCELoss(NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "alexnet_unet_best.pt"
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * x.size(0)

        iou, dice = evaluate(model, val_dl, device, NUM_CLASSES)
        miou, mdice = iou.nanmean().item(), dice.nanmean().item()
        mark = ""
        if miou > best:
            best, mark = miou, " *"
            torch.save(model.state_dict(), ckpt)
        print(f"epoch {epoch:02d} | train loss {total / len(train_dl.dataset):.4f} | val mIoU {miou:.4f} | val Dice {mdice:.4f}{mark}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    iou, dice = evaluate(model, test_dl, device, NUM_CLASSES)
    print(f"\nTEST mIoU {iou.nanmean():.4f} | Dice {dice.nanmean():.4f}")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:<10s}: IoU {iou[c]:.4f} | Dice {dice[c]:.4f}")


if __name__ == "__main__":
    main()