# AlexNet U-Net segmentation report

- Checkpoint: `src/cnn_segmentation/models/checkpoints_4cls/alexnet_unet_best.pt`
- Parameters: 3,420,660

## Overall

| Split | Images | Pixel Acc | mIoU | mDice |
|---|---|---|---|---|
| train | 1749 | 0.9936 | 0.9183 | 0.9566 |
| val | 375 | 0.9890 | 0.8643 | 0.9246 |
| test | 375 | 0.9898 | 0.8747 | 0.9310 |

## Per-class (val)

| Class | IoU | Dice | Recall |
|---|---|---|---|
| Background | 0.9890 | 0.9945 | 0.9938 |
| PE | 0.7699 | 0.8700 | 0.8856 |
| CC | 0.7827 | 0.8781 | 0.8855 |
| IVD | 0.9156 | 0.9559 | 0.9594 |

## Per-class (test)

| Class | IoU | Dice | Recall |
|---|---|---|---|
| Background | 0.9898 | 0.9949 | 0.9948 |
| PE | 0.7837 | 0.8787 | 0.8878 |
| CC | 0.8033 | 0.8909 | 0.8914 |
| IVD | 0.9222 | 0.9595 | 0.9550 |

## AlexNet vs LeNet (test)

| Metric | LeNet | AlexNet |
|---|---|---|
| Pixel Acc | 0.9890 | 0.9898 |
| mIoU | 0.8680 | 0.8747 |
| mDice | 0.9270 | 0.9310 |
| IoU Background | 0.9890 | 0.9898 |
| Dice Background | 0.9944 | 0.9949 |
| IoU PE | 0.7695 | 0.7837 |
| Dice PE | 0.8698 | 0.8787 |
| IoU CC | 0.7987 | 0.8033 |
| Dice CC | 0.8881 | 0.8909 |
| IoU IVD | 0.9149 | 0.9222 |
| Dice IVD | 0.9556 | 0.9595 |
