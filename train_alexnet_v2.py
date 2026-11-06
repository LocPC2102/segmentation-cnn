"""Train AlexNetUNetV2 with AdamW, LR reduction on plateau and early stopping.

Run from the project root:
    python -m src.cnn_segmentation.train_alexnet_v2
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
from src.cnn_segmentation.models.alexnet_unet_v2 import AlexNetUNetV2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=15, help="stop after this many epochs without val mIoU gain")
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

    model = AlexNetUNetV2(NUM_CLASSES, args.dropout).to(device)
    criterion = DiceCELoss(NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "alexnet_unet_v2_best.pt"
    best, stale = -1.0, 0
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        scheduler.step(miou)

        mark = ""
        if miou > best:
            best, stale, mark = miou, 0, " *"
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1
        print(f"epoch {epoch:02d} | lr {optimizer.param_groups[0]['lr']:.1e} | train loss {total / len(train_dl.dataset):.4f}"
              f" | val mIoU {miou:.4f} | val Dice {mdice:.4f}{mark}")
        if stale >= args.patience:
            print(f"early stop: no gain for {args.patience} epochs")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    iou, dice = evaluate(model, test_dl, device, NUM_CLASSES)
    print(f"\nTEST mIoU {iou.nanmean():.4f} | Dice {dice.nanmean():.4f}")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:<10s}: IoU {iou[c]:.4f} | Dice {dice[c]:.4f}")


if __name__ == "__main__":
    main()