GPU: NVIDIA L4 · torch 2.14.1+cu130 · timm 1.0.31 · random inputs, mixed precision

| Model | Image size | Params (M) | GFLOPs | Inference (img/s) | Train (img/s) | Epoch at 100% (min) | Peak train GPU (GB) |
|---|---|---|---|---|---|---|---|
| small_cnn_scratch | 32 | 1.15 | 0.31 | 69809 | 5641 | 0.1 | 0.11 |
| resnet50_finetune | 224 | 23.53 | 8.17 | 1222 | 428 | 1.8 | 1.68 |
| convnext_tiny_finetune | 224 | 27.83 | 8.91 | 980 | 371 | 2.0 | 2.26 |
| small_vit_scratch_noclip | 32 | 1.21 | 0.17 | 55771 | 1912 | 0.4 | 0.10 |
| vit_small_finetune_noclip | 224 | 21.67 | 9.20 | 1768 | 636 | 1.2 | 1.37 |
