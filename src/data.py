import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")


def cifar_data_config(image_size=32):
    return {
        "input_size": (3, image_size, image_size),
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
        "interpolation": "bilinear",
    }


def load_raw_datasets(cfg, train_transform=None, test_transform=None):
    data_cfg = cfg["data"]
    if data_cfg["name"] != "cifar10":
        raise ValueError(f"Unsupported dataset: {data_cfg['name']}")
    root = data_cfg["root"]
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
    return train_set, test_set


def make_train_val_indices(labels, val_fraction, seed):
    labels = np.asarray(labels)
    all_indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        all_indices, test_size=val_fraction, stratify=labels, random_state=seed
    )
    return np.sort(train_idx), np.sort(val_idx)


def subsample_train_indices(train_indices, labels, fraction, seed):
    train_indices = np.asarray(train_indices)
    if fraction >= 1.0:
        return train_indices.copy()
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(labels[train_indices]):
        cls_indices = np.sort(train_indices[labels[train_indices] == cls])
        shuffled = rng.permutation(cls_indices)
        n_keep = max(1, int(round(fraction * len(cls_indices))))
        chosen.append(shuffled[:n_keep])
    return np.sort(np.concatenate(chosen))


def _interpolation(name):
    modes = {
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "nearest": transforms.InterpolationMode.NEAREST,
    }
    return modes.get(name, transforms.InterpolationMode.BILINEAR)


def build_transforms(data_cfg, model_data_cfg, augmentation, is_train):
    image_size = model_data_cfg["input_size"][-1]
    ops = []

    if is_train and augmentation != "none":
        if augmentation not in ("basic", "strong"):
            raise ValueError(f"Unknown augmentation level: {augmentation}")
        ops += [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ]
        if augmentation == "strong":
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))

    if image_size != 32:
        ops.append(transforms.Resize(image_size,
                                     interpolation=_interpolation(model_data_cfg.get("interpolation")),
                                     antialias=True))

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(model_data_cfg["mean"], model_data_cfg["std"]),
    ]
    return transforms.Compose(ops)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_args(batch_size, num_workers, persistent):
    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        args.update(persistent_workers=persistent, worker_init_fn=_seed_worker, timeout=300)
    return args


def release_workers(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()
    loader._iterator = None


def get_split_indices(cfg, labels):
    data_cfg = cfg["data"]
    train_idx, val_idx = make_train_val_indices(
        labels, data_cfg["val_fraction"], data_cfg.get("split_seed", 0)
    )
    train_idx = subsample_train_indices(train_idx, labels, data_cfg["train_fraction"], cfg["seed"])
    return train_idx, val_idx


def build_dataloaders(cfg, model_data_cfg):
    data_cfg = cfg["data"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = data_cfg["num_workers"]

    train_tf = build_transforms(data_cfg, model_data_cfg, data_cfg["augmentation"], is_train=True)
    eval_tf = build_transforms(data_cfg, model_data_cfg, "none", is_train=False)

    train_full, test_set = load_raw_datasets(cfg, train_transform=train_tf, test_transform=eval_tf)
    val_full, _ = load_raw_datasets(cfg, train_transform=eval_tf, test_transform=eval_tf)

    train_idx, val_idx = get_split_indices(cfg, train_full.targets)
    train_subset = Subset(train_full, train_idx.tolist())
    val_subset = Subset(val_full, val_idx.tolist())

    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_subset, shuffle=True, drop_last=len(train_subset) > batch_size,
        generator=generator, **_loader_args(batch_size, num_workers, persistent=True)
    )

    eval_workers = data_cfg.get("eval_num_workers", min(2, num_workers))
    val_loader = DataLoader(val_subset, shuffle=False,
                            **_loader_args(batch_size, eval_workers, persistent=True))
    test_loader = DataLoader(test_set, shuffle=False,
                             **_loader_args(batch_size, eval_workers, persistent=False))

    num_classes = len(train_full.classes)
    return train_loader, val_loader, test_loader, num_classes
