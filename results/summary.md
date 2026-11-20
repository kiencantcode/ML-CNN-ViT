| Model | Fraction | Aug | Seeds | Test acc (%) | Macro-F1 (%) | Params (M) | Train time (min) | Peak GPU (GB) |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny (pretrained) | 0.01 | basic | 3 | 92.14 ± 0.61 | 92.13 ± 0.62 | 27.83 | 3.5 | 2.26 |
| ConvNeXt-Tiny (pretrained) | 0.1 | basic | 3 | 96.69 ± 0.19 | 96.68 ± 0.20 | 27.83 | 7.5 | 2.26 |
| ConvNeXt-Tiny (pretrained) | 1 | basic | 3 | 98.13 ± 0.09 | 98.13 ± 0.09 | 27.83 | 42.4 | 2.26 |
| ConvNeXt-Tiny (pretrained) | 0.1 | none | 3 | 96.35 ± 0.12 | 96.35 ± 0.12 | 27.83 | 6.8 | 2.26 |
| ConvNeXt-Tiny (pretrained) | 0.1 | strong | 3 | 96.68 ± 0.22 | 96.68 ± 0.21 | 27.83 | 6.8 | 2.26 |
| ConvNeXt-Tiny (frozen backbone) [frozen] | 0.1 | basic | 3 | 91.94 ± 0.12 | 91.92 ± 0.12 | 27.83 | 4.2 | 0.39 |
| ConvNeXt-Tiny (frozen backbone) [frozen] | 1 | basic | 3 | 93.94 ± 0.04 | 93.93 ± 0.04 | 27.83 | 16.2 | 0.39 |
| ResNet-50 (pretrained) | 0.01 | basic | 3 | 73.62 ± 0.35 | 73.44 ± 0.31 | 23.53 | 3.3 | 1.69 |
| ResNet-50 (pretrained) | 0.1 | basic | 3 | 92.70 ± 0.25 | 92.68 ± 0.26 | 23.53 | 6.3 | 1.69 |
| ResNet-50 (pretrained) | 1 | basic | 3 | 97.06 ± 0.14 | 97.06 ± 0.14 | 23.53 | 39.8 | 1.69 |
| ResNet-50 (pretrained) | 0.1 | none | 3 | 92.12 ± 0.35 | 92.11 ± 0.35 | 23.53 | 6.0 | 1.69 |
| ResNet-50 (pretrained) | 0.1 | strong | 3 | 92.81 ± 0.21 | 92.78 ± 0.22 | 23.53 | 6.0 | 1.69 |
| ResNet-50 (frozen backbone) [frozen] | 0.1 | basic | 3 | 88.04 ± 0.17 | 87.98 ± 0.19 | 23.53 | 3.8 | 0.43 |
| ResNet-50 (frozen backbone) [frozen] | 1 | basic | 3 | 91.74 ± 0.07 | 91.71 ± 0.07 | 23.53 | 15.6 | 0.43 |
| ResNet-50 (random init) | 0.1 | basic | 3 | 56.57 ± 2.14 | 56.04 ± 2.15 | 23.53 | 6.9 | 1.68 |
| ResNet-50 (random init) | 1 | basic | 3 | 85.44 ± 0.08 | 85.39 ± 0.06 | 23.53 | 38.4 | 1.68 |
| SmallCNN (scratch) | 0.01 | basic | 3 | 49.57 ± 1.35 | 49.65 ± 1.28 | 1.15 | 2.2 | 0.08 |
| SmallCNN (scratch) | 0.1 | basic | 3 | 77.99 ± 0.52 | 77.95 ± 0.53 | 1.15 | 3.5 | 0.08 |
| SmallCNN (scratch) | 1 | basic | 3 | 91.84 ± 0.13 | 91.82 ± 0.12 | 1.15 | 16.7 | 0.08 |
| SmallCNN (scratch) | 0.1 | none | 3 | 71.77 ± 0.75 | 71.69 ± 0.76 | 1.15 | 3.1 | 0.08 |
| SmallCNN (scratch) | 0.1 | strong | 3 | 79.73 ± 0.47 | 79.67 ± 0.49 | 1.15 | 3.3 | 0.08 |
