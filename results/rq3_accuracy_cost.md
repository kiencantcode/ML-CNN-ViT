Accuracy: test set, mean ± std over seeds, 100% data, basic augmentation. Cost: one NVIDIA L4.

| Model | Input | Params (M) | GFLOPs | Inference (img/s) | Train (img/s) | Epoch at 100% (min) | Peak train GPU (GB) | Test acc (%) |
|---|---|---|---|---|---|---|---|---|
| SmallCNN (scratch) | 32 | 1.15 | 0.31 | 69,809 | 5,641 | 0.1 | 0.11 | 91.84 ± 0.13 |
| ResNet-50 (pretrained) | 224 | 23.53 | 8.17 | 1,222 | 428 | 1.8 | 1.68 | 97.06 ± 0.14 |
| ConvNeXt-Tiny (pretrained) | 224 | 27.83 | 8.91 | 980 | 371 | 2.0 | 2.26 | 98.13 ± 0.09 |
| SmallViT (scratch) | 32 | 1.21 | 0.17 | 55,771 | 1,912 | 0.4 | 0.10 | 81.84 ± 0.13 |
| DeiT-S (pretrained) | 224 | 21.67 | 9.20 | 1,768 | 636 | 1.2 | 1.37 | 98.08 ± 0.11 |
