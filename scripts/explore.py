# %% [markdown]
# # Explore the data and models
# Run each cell with "Run Cell" (or Shift+Enter) in VS Code.
# Figures are also saved to outputs/explore/.

# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import (CIFAR10_CLASSES, build_dataloaders, build_transforms,
                      cifar_data_config, get_split_indices, load_raw_datasets)
from src.models import build_model, get_model_data_config, get_param_groups
from src.utils import count_parameters, load_config, set_seed

OUT = ROOT / "outputs" / "explore"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)


def save(fig, name):
    fig.savefig(OUT / name, dpi=120, bbox_inches="tight")
    plt.show()


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# %% [markdown]
# ## 1. Config: what one run will use

# %%
cfg = load_config(ROOT / "configs/small_cnn_scratch.yaml")
cfg["data"]["root"] = str(ROOT / cfg["data"]["root"])
cfg["data"]["num_workers"] = 0
for section in ["data", "model", "train"]:
    print(section, cfg[section])

# %% [markdown]
# ## 2. Raw CIFAR-10 images

# %%
train_set, test_set = load_raw_datasets(cfg)
labels = np.array(train_set.targets)
print("train images:", train_set.data.shape, " test images:", test_set.data.shape)
print("images per class:", np.bincount(labels))

fig, axes = plt.subplots(4, 8, figsize=(12, 6.5))
for ax, idx in zip(axes.flat, np.random.default_rng(0).choice(len(train_set), 32, replace=False)):
    ax.imshow(train_set.data[idx])
    ax.set_title(CIFAR10_CLASSES[labels[idx]], fontsize=9)
    ax.axis("off")
fig.suptitle("CIFAR-10 training images (32 × 32)")
save(fig, "1_cifar_samples.png")

# %% [markdown]
# ## 3. Split and learning-curve subsets

# %%
for fraction in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
    cfg_f = {**cfg, "data": {**cfg["data"], "train_fraction": fraction}}
    train_idx, val_idx = get_split_indices(cfg_f, labels)
    print(f"fraction {fraction:>4}: train {len(train_idx):>5}, per class {np.bincount(labels[train_idx])[0]:>4}, val {len(val_idx)}")

# %% [markdown]
# ## 4. Augmentation levels on the same image
# Each column is a new random draw. "none" never changes.

# %%
image = train_set[7][0]
cifar_cfg = cifar_data_config(32)
levels = ["none", "basic", "strong"]
fig, axes = plt.subplots(len(levels), 6, figsize=(10, 5.4))
for row, level in enumerate(levels):
    tf = build_transforms(cfg["data"], cifar_cfg, level, is_train=True)
    for col in range(6):
        ax = axes[row, col]
        ax.imshow(denormalize(tf(image), cifar_cfg["mean"], cifar_cfg["std"]))
        ax.set_xticks([]), ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(level, fontsize=11)
fig.suptitle(f"Augmentation levels ({CIFAR10_CLASSES[train_set.targets[7]]})")
save(fig, "2_augmentation.png")

# %% [markdown]
# ## 5. What each model receives
# SmallCNN sees 32 × 32. Pretrained models see the same image resized to 224 × 224.

# %%
resnet_cfg = load_config(ROOT / "configs/resnet50_finetune.yaml")
resnet = build_model({**resnet_cfg["model"], "pretrained": False}, 10)
resnet_data = get_model_data_config(resnet, resnet_cfg["model"], resnet_cfg["data"])
print("ResNet-50 data config:", resnet_data)

small_x = build_transforms(cfg["data"], cifar_cfg, "none", is_train=False)(image)
big_x = build_transforms(resnet_cfg["data"], resnet_data, "none", is_train=False)(image)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(denormalize(small_x, cifar_cfg["mean"], cifar_cfg["std"]))
axes[0].set_title(f"SmallCNN input {tuple(small_x.shape)}")
axes[1].imshow(denormalize(big_x, resnet_data["mean"], resnet_data["std"]))
axes[1].set_title(f"ResNet-50 input {tuple(big_x.shape)}")
for ax in axes:
    ax.axis("off")
save(fig, "3_model_inputs.png")

# %% [markdown]
# ## 6. Inside SmallCNN: layers, parameters, shapes

# %%
model = build_model(cfg["model"], num_classes=10).to(DEVICE)
print(model)
total, trainable = count_parameters(model)
print(f"\nparameters: {total:,} total, {trainable:,} trainable")

x = small_x.unsqueeze(0).to(DEVICE)
print("\ninput  ", tuple(x.shape))
with torch.no_grad():
    for name in ["stage1", "stage2", "stage3"]:
        x = getattr(model, name)(x)
        print(f"{name} ", tuple(x.shape))
    pooled = model.pool(x).flatten(1)
    print("pooled ", tuple(pooled.shape))
    print("logits ", tuple(model.fc(pooled).shape))

# %% [markdown]
# ## 7. Feature maps after stage 1 (untrained network)
# Each small image is one of the 64 channels. After training these become edge and colour detectors.

# %%
model.eval()
with torch.no_grad():
    maps = model.stage1(small_x.unsqueeze(0).to(DEVICE))[0].cpu()
fig, axes = plt.subplots(4, 8, figsize=(10, 5.4))
for i, ax in enumerate(axes.flat):
    ax.imshow(maps[i], cmap="viridis")
    ax.set_title(f"ch {i}", fontsize=8)
    ax.axis("off")
fig.suptitle("SmallCNN stage 1 feature maps (first 32 of 64, 16 × 16 each)")
save(fig, "4_feature_maps.png")

# %% [markdown]
# ## 8. Predictions before training
# An untrained model is close to random guessing: about 10% accuracy, nearly flat probabilities.

# %%
train_loader, val_loader, test_loader, num_classes = build_dataloaders(cfg, cifar_cfg)
images, targets = next(iter(val_loader))
with torch.no_grad():
    probs = model(images.to(DEVICE)).softmax(dim=1).cpu()
accuracy = (probs.argmax(1) == targets).float().mean().item()
print(f"accuracy on one validation batch before training: {accuracy:.1%}")

fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(CIFAR10_CLASSES, probs[0].numpy())
ax.set_ylabel("probability")
ax.set_title(f"Untrained SmallCNN on one image (true class: {CIFAR10_CLASSES[targets[0]]})")
ax.tick_params(axis="x", rotation=45)
save(fig, "5_untrained_prediction.png")

# %% [markdown]
# ## 9. Parameter groups the optimizer will receive

# %%
for name, path in [("SmallCNN", "configs/small_cnn_scratch.yaml"),
                   ("ResNet-50", "configs/resnet50_finetune.yaml")]:
    c = load_config(ROOT / path)
    m = model if name == "SmallCNN" else resnet
    print(name)
    for g in get_param_groups(m, c["train"]["lr_head"], c["train"]["lr_backbone"], c["train"]["weight_decay"]):
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:<18} {len(g['params']):>4} tensors {n:>11,} values  lr {g['lr']:g}  wd {g['weight_decay']}")
