| Model | Fraction | Aug | Seeds | Test acc (%) | Macro-F1 (%) | Params (M) | Train time (min) | Peak GPU (GB) |
|---|---|---|---|---|---|---|---|---|
| SmallViT + grad clip (ablation) | 0.01 | basic | 3 | 31.84 ± 0.60 | 31.60 ± 0.69 | 1.21 | 1.3 | 0.10 |
| SmallViT + grad clip (ablation) | 0.1 | basic | 3 | 53.08 ± 0.86 | 52.81 ± 0.86 | 1.21 | 3.8 | 0.10 |
| SmallViT + grad clip (ablation) | 1 | basic | 3 | 80.99 ± 0.50 | 80.94 ± 0.48 | 1.21 | 29.2 | 0.10 |
| SmallViT (scratch, lr 1e-3) | 0.01 | basic | 3 | 32.12 ± 0.27 | 31.89 ± 0.32 | 1.21 | 1.3 | 0.10 |
| SmallViT (scratch, lr 1e-3) | 0.1 | basic | 3 | 54.58 ± 1.61 | 54.34 ± 1.66 | 1.21 | 3.7 | 0.10 |
| SmallViT (scratch, lr 1e-3) | 1 | basic | 3 | 80.99 ± 0.47 | 80.91 ± 0.49 | 1.21 | 27.4 | 0.10 |
| SmallViT (scratch, lr 1e-3) | 0.1 | none | 3 | 45.73 ± 0.30 | 45.24 ± 0.46 | 1.21 | 3.4 | 0.10 |
| SmallViT (scratch, lr 1e-3) | 0.1 | strong | 3 | 54.27 ± 0.95 | 53.94 ± 0.97 | 1.21 | 3.7 | 0.10 |
| SmallViT (scratch) | 0.01 | basic | 3 | 33.21 ± 0.62 | 33.06 ± 0.77 | 1.21 | 1.3 | 0.10 |
| SmallViT (scratch) | 0.1 | basic | 3 | 56.40 ± 0.11 | 56.25 ± 0.23 | 1.21 | 3.6 | 0.10 |
| SmallViT (scratch) | 1 | basic | 3 | 81.84 ± 0.13 | 81.78 ± 0.14 | 1.21 | 27.0 | 0.10 |
| SmallViT (scratch) | 0.1 | none | 3 | 47.82 ± 0.21 | 47.16 ± 0.51 | 1.21 | 3.4 | 0.10 |
| SmallViT (scratch) | 0.1 | strong | 3 | 58.68 ± 0.67 | 58.39 ± 0.58 | 1.21 | 3.6 | 0.10 |
| DeiT-S + grad clip (ablation) | 0.01 | basic | 3 | 91.97 ± 0.30 | 91.94 ± 0.29 | 21.67 | 0.9 | 1.37 |
| DeiT-S + grad clip (ablation) | 0.05 | basic | 3 | 94.99 ± 0.23 | 94.98 ± 0.23 | 21.67 | 1.4 | 1.37 |
| DeiT-S + grad clip (ablation) | 0.1 | basic | 3 | 95.79 ± 0.11 | 95.78 ± 0.11 | 21.67 | 1.9 | 1.37 |
| DeiT-S + grad clip (ablation) | 0.25 | basic | 3 | 96.85 ± 0.19 | 96.85 ± 0.19 | 21.67 | 3.3 | 1.37 |
| DeiT-S + grad clip (ablation) | 0.5 | basic | 3 | 97.39 ± 0.15 | 97.39 ± 0.15 | 21.67 | 5.7 | 1.37 |
| DeiT-S + grad clip (ablation) | 1 | basic | 3 | 97.91 ± 0.06 | 97.91 ± 0.06 | 21.67 | 10.6 | 1.37 |
| DeiT-S (pretrained, lr 1e-4) | 0.01 | basic | 3 | 91.22 ± 0.12 | 91.22 ± 0.11 | 21.67 | 1.0 | 1.37 |
| DeiT-S (pretrained, lr 1e-4) | 0.05 | basic | 3 | 94.99 ± 0.33 | 94.98 ± 0.33 | 21.67 | 1.4 | 1.37 |
| DeiT-S (pretrained, lr 1e-4) | 0.1 | basic | 3 | 95.84 ± 0.18 | 95.83 ± 0.17 | 21.67 | 1.9 | 1.37 |
| DeiT-S (pretrained, lr 1e-4) | 0.25 | basic | 3 | 96.87 ± 0.19 | 96.86 ± 0.19 | 21.67 | 3.2 | 1.37 |
| DeiT-S (pretrained, lr 1e-4) | 0.5 | basic | 3 | 97.36 ± 0.13 | 97.36 ± 0.13 | 21.67 | 5.6 | 1.37 |
| DeiT-S (pretrained, lr 1e-4) | 1 | basic | 3 | 97.78 ± 0.13 | 97.78 ± 0.13 | 21.67 | 10.6 | 1.45 |
| DeiT-S (pretrained, lr 1e-4) | 0.1 | none | 3 | 95.37 ± 0.22 | 95.37 ± 0.21 | 21.67 | 1.9 | 1.37 |
| DeiT-S (pretrained, lr 1e-4) | 0.1 | strong | 3 | 96.00 ± 0.26 | 96.00 ± 0.26 | 21.67 | 1.8 | 1.37 |
| DeiT-S (pretrained) | 0.01 | basic | 3 | 92.61 ± 0.33 | 92.60 ± 0.34 | 21.67 | 0.9 | 1.37 |
| DeiT-S (pretrained) | 0.05 | basic | 3 | 95.71 ± 0.10 | 95.71 ± 0.09 | 21.67 | 1.4 | 1.37 |
| DeiT-S (pretrained) | 0.1 | basic | 3 | 96.29 ± 0.19 | 96.29 ± 0.19 | 21.67 | 1.9 | 1.37 |
| DeiT-S (pretrained) | 0.25 | basic | 3 | 97.36 ± 0.04 | 97.36 ± 0.04 | 21.67 | 3.2 | 1.37 |
| DeiT-S (pretrained) | 0.5 | basic | 3 | 97.68 ± 0.11 | 97.68 ± 0.11 | 21.67 | 5.6 | 1.37 |
| DeiT-S (pretrained) | 1 | basic | 3 | 98.08 ± 0.11 | 98.08 ± 0.11 | 21.67 | 10.5 | 1.37 |
| DeiT-S (pretrained) | 0.1 | none | 3 | 96.14 ± 0.13 | 96.14 ± 0.13 | 21.67 | 1.9 | 1.37 |
| DeiT-S (pretrained) | 0.1 | strong | 3 | 96.38 ± 0.26 | 96.38 ± 0.25 | 21.67 | 1.9 | 1.37 |
| DeiT-S (frozen backbone) [frozen] | 0.1 | basic | 3 | 92.60 ± 0.24 | 92.60 ± 0.24 | 21.67 | 1.1 | 0.19 |
| DeiT-S (frozen backbone) [frozen] | 1 | basic | 3 | 93.87 ± 0.06 | 93.87 ± 0.06 | 21.67 | 4.3 | 0.19 |
| ViT-S/16 (random init, lr 1e-3) | 0.1 | basic | 3 | 35.99 ± 1.25 | 35.26 ± 1.48 | 21.67 | 1.9 | 1.37 |
| ViT-S/16 (random init, lr 1e-3) | 1 | basic | 3 | 32.61 ± 1.61 | 31.71 ± 1.77 | 21.67 | 10.5 | 1.37 |
| ViT-S/16 (random init) | 0.1 | basic | 3 | 44.97 ± 0.46 | 44.43 ± 0.45 | 21.67 | 1.9 | 1.37 |
| ViT-S/16 (random init) | 1 | basic | 3 | 65.38 ± 0.54 | 65.17 ± 0.58 | 21.67 | 10.6 | 1.37 |
