Test accuracy per class (%), mean ± std over 3 seeds, 100% data, basic augmentation. 1,000 test images per class (1 image = 0.1 points).

| Class | SmallCNN | ResNet-50 | ConvNeXt-T | SmallViT | DeiT-S |
|---|---|---|---|---|---|
| airplane | 93.1 ± 0.5 | 97.3 ± 0.7 | 99.1 ± 0.1 | 82.7 ± 0.5 | 98.5 ± 0.5 |
| automobile | 96.4 ± 0.4 | 98.5 ± 0.4 | 99.0 ± 0.1 | 91.0 ± 0.3 | 99.1 ± 0.1 |
| bird | 88.3 ± 0.4 | 97.3 ± 0.6 | 98.1 ± 0.2 | 75.3 ± 1.0 | 97.8 ± 0.2 |
| cat | 82.3 ± 0.4 | 93.2 ± 0.7 | 95.5 ± 0.4 | 64.9 ± 1.4 | 95.7 ± 0.4 |
| deer | 92.0 ± 0.4 | 96.4 ± 0.7 | 97.9 ± 0.2 | 77.8 ± 0.8 | 97.7 ± 0.3 |
| dog | 87.5 ± 0.4 | 96.2 ± 0.3 | 96.5 ± 0.1 | 76.9 ± 1.4 | 97.2 ± 0.6 |
| frog | 94.6 ± 1.0 | 98.5 ± 1.0 | 99.3 ± 0.2 | 88.3 ± 0.7 | 99.3 ± 0.4 |
| horse | 94.2 ± 0.9 | 97.4 ± 0.5 | 98.6 ± 0.1 | 84.0 ± 0.2 | 98.2 ± 0.6 |
| ship | 95.9 ± 0.1 | 98.5 ± 0.0 | 99.2 ± 0.0 | 88.8 ± 0.7 | 99.3 ± 0.3 |
| truck | 94.0 ± 0.6 | 97.4 ± 0.3 | 98.0 ± 0.6 | 88.6 ± 0.1 | 97.9 ± 0.2 |

Most confused pairs (both directions, errors per seed):

| Model | Pair | a → b | b → a | Total |
|---|---|---|---|---|
| SmallCNN | cat ↔ dog | 81.3 | 65.0 | 146.3 |
| SmallCNN | automobile ↔ truck | 23.7 | 30.7 | 54.3 |
| SmallCNN | bird ↔ deer | 25.0 | 20.3 | 45.3 |
| SmallCNN | airplane ↔ ship | 23.7 | 20.7 | 44.3 |
| SmallCNN | bird ↔ frog | 24.0 | 18.0 | 42.0 |
| ResNet-50 | cat ↔ dog | 43.3 | 28.0 | 71.3 |
| ResNet-50 | automobile ↔ truck | 13.0 | 20.7 | 33.7 |
| ResNet-50 | airplane ↔ ship | 10.7 | 10.7 | 21.3 |
| ResNet-50 | deer ↔ horse | 9.3 | 9.3 | 18.7 |
| ResNet-50 | cat ↔ deer | 7.3 | 11.0 | 18.3 |
| ConvNeXt-T | cat ↔ dog | 31.3 | 30.0 | 61.3 |
| ConvNeXt-T | automobile ↔ truck | 9.0 | 17.7 | 26.7 |
| ConvNeXt-T | cat ↔ deer | 3.3 | 7.3 | 10.7 |
| ConvNeXt-T | airplane ↔ ship | 4.0 | 5.7 | 9.7 |
| ConvNeXt-T | bird ↔ deer | 4.0 | 5.0 | 9.0 |
| SmallViT | cat ↔ dog | 159.3 | 124.3 | 283.7 |
| SmallViT | bird ↔ deer | 52.3 | 53.3 | 105.7 |
| SmallViT | automobile ↔ truck | 54.7 | 51.0 | 105.7 |
| SmallViT | deer ↔ horse | 49.3 | 45.3 | 94.7 |
| SmallViT | airplane ↔ ship | 45.3 | 48.0 | 93.3 |
| DeiT-S | cat ↔ dog | 27.7 | 21.7 | 49.3 |
| DeiT-S | automobile ↔ truck | 8.0 | 18.7 | 26.7 |
| DeiT-S | airplane ↔ ship | 10.0 | 4.3 | 14.3 |
| DeiT-S | deer ↔ horse | 7.3 | 4.7 | 12.0 |
| DeiT-S | cat ↔ deer | 3.0 | 7.7 | 10.7 |
