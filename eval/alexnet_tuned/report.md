# AlexNet U-Net (tuned) segmentation report

- Checkpoint: `src/cnn_segmentation/models/checkpoints_4cls/alexnet_unet_tuned_best.pt`
- Parameters: 3,420,660

## Overall

| Split | Images | Pixel Acc | mIoU | mDice |
|---|---|---|---|---|
| train | 1749 | 0.9922 | 0.8999 | 0.9460 |
| val | 375 | 0.9895 | 0.8690 | 0.9275 |
| test | 375 | 0.9900 | 0.8759 | 0.9317 |

## Per-class (val)

| Class | IoU | Dice | Recall |
|---|---|---|---|
| Background | 0.9895 | 0.9947 | 0.9942 |
| PE | 0.7773 | 0.8747 | 0.8789 |
| CC | 0.7903 | 0.8829 | 0.8893 |
| IVD | 0.9189 | 0.9577 | 0.9640 |

## Per-class (test)

| Class | IoU | Dice | Recall |
|---|---|---|---|
| Background | 0.9899 | 0.9949 | 0.9953 |
| PE | 0.7858 | 0.8801 | 0.8738 |
| CC | 0.8059 | 0.8925 | 0.8927 |
| IVD | 0.9219 | 0.9594 | 0.9563 |

## Comparison (test)

| Metric | LeNet | AlexNet | AlexNet tuned |
|---|---|---|---|
| Pixel Acc | 0.9890 | 0.9898 | 0.9900 |
| mIoU | 0.8680 | 0.8747 | 0.8759 |
| mDice | 0.9270 | 0.9310 | 0.9317 |
| IoU Background | 0.9890 | 0.9898 | 0.9899 |
| Dice Background | 0.9944 | 0.9949 | 0.9949 |
| IoU PE | 0.7695 | 0.7837 | 0.7858 |
| Dice PE | 0.8698 | 0.8787 | 0.8801 |
| IoU CC | 0.7987 | 0.8033 | 0.8059 |
| Dice CC | 0.8881 | 0.8909 | 0.8925 |
| IoU IVD | 0.9149 | 0.9222 | 0.9219 |
| Dice IVD | 0.9556 | 0.9595 | 0.9594 |
