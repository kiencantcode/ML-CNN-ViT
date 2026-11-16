import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CIFAR10_CLASSES, build_transforms, load_raw_datasets
from src.engine import autocast
from src.models import build_model, get_model_data_config
from src.utils import get_hardware_info, load_config

GROUPS = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
    "weather": ["snow", "frost", "fog", "brightness", "spatter"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression", "saturate"],
}
CORRUPTIONS = [c for group in GROUPS.values() for c in group]
SEVERITIES = [1, 2, 3, 4, 5]


class CorruptedImages(Dataset):
    def __init__(self, folder, items, clean_images, transform):
        self.folder = str(folder)
        self.items = items
        self.clean_images = clean_images
        self.transform = transform
        self.arrays = {}

    def __len__(self):
        return len(self.items)

    def _array(self, name):
        if name not in self.arrays:
            self.arrays[name] = np.load(f"{self.folder}/{name}.npy", mmap_mode="r", allow_pickle=False)
        return self.arrays[name]

    def __getitem__(self, i):
        corruption, severity, index = self.items[i]
        if corruption == "clean":
            image = self.clean_images[index]
        else:
            image = self._array(corruption)[(severity - 1) * 10000 + index]
        return self.transform(Image.fromarray(np.ascontiguousarray(image))), i


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models on CIFAR-10-C")
    parser.add_argument("--data", default="data/CIFAR-10-C")
    parser.add_argument("--experiments", nargs="+",
                        default=["resnet50_finetune", "convnext_tiny_finetune", "small_cnn_scratch"])
    parser.add_argument("--runs", nargs="+", default=["frac1.0_augbasic_seed0"])
    parser.add_argument("--per-class", type=int, default=200,
                        help="test images per class used for every corruption and severity")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default="results/cifar10c")
    return parser.parse_args()


def fixed_subset(labels, per_class):
    rng = np.random.default_rng(0)
    picked = [np.sort(rng.choice(np.flatnonzero(labels == c), per_class, replace=False))
              for c in range(len(CIFAR10_CLASSES))]
    return np.sort(np.concatenate(picked))


def evaluate(model, loader, items, labels, device):
    correct = np.zeros(len(items), dtype=bool)
    model.eval()
    with torch.no_grad():
        for images, positions in loader:
            with autocast(device, True):
                preds = model(images.to(device, non_blocking=True)).argmax(1).cpu().numpy()
            positions = positions.numpy()
            truth = np.array([labels[items[p][2]] for p in positions])
            correct[positions] = preds == truth
    return correct


def main():
    args = parse_args()
    folder = ROOT / args.data
    if not (folder / "labels.npy").exists():
        print(f"CIFAR-10-C not found in {folder}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg = load_config(ROOT / "configs" / "base.yaml")
    base_cfg["data"]["root"] = str(ROOT / base_cfg["data"]["root"])
    _, test_set = load_raw_datasets(base_cfg)
    labels = np.asarray(test_set.targets)
    c_labels = np.load(folder / "labels.npy", allow_pickle=False)[:10000]
    assert np.array_equal(c_labels, labels), "CIFAR-10-C labels do not match the CIFAR-10 test set"

    subset = fixed_subset(labels, args.per_class)
    items = [("clean", 0, int(i)) for i in subset]
    items += [(c, s, int(i)) for c in CORRUPTIONS for s in SEVERITIES for i in subset]
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(subset)} test images × (1 clean + {len(CORRUPTIONS)} corruptions × 5 severities) "
          f"= {len(items):,} images per model")

    for experiment in args.experiments:
        for run in args.runs:
            run_dir = ROOT / "outputs" / experiment / run
            out_file = out_dir / f"{experiment}_{run}.json"
            if not (run_dir / "best.pt").exists():
                print(f"skip {experiment}/{run}: no checkpoint")
                continue
            if out_file.exists():
                print(f"skip {experiment}/{run}: already evaluated")
                continue

            cfg = load_config(run_dir / "config.yaml")
            model = build_model({**cfg["model"], "pretrained": False}, len(CIFAR10_CLASSES))
            model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
            model.to(device)
            data_cfg = get_model_data_config(model, cfg["model"], cfg["data"])
            transform = build_transforms(cfg["data"], data_cfg, "none", is_train=False)
            dataset = CorruptedImages(folder, items, test_set.data, transform)
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                pin_memory=device.type == "cuda",
                                timeout=600 if args.workers else 0)

            start = time.perf_counter()
            correct = evaluate(model, loader, items, labels, device)
            minutes = (time.perf_counter() - start) / 60

            keys = [(c, s) for c, s, _ in items]
            per = {}
            for (c, s), ok in zip(keys, correct):
                per.setdefault(f"{c}/{s}", []).append(ok)
            acc = {k: float(np.mean(v)) for k, v in per.items()}
            clean = acc.pop("clean/0")
            by_corruption = {c: float(np.mean([acc[f"{c}/{s}"] for s in SEVERITIES])) for c in CORRUPTIONS}
            by_group = {g: float(np.mean([by_corruption[c] for c in cs])) for g, cs in GROUPS.items()}
            by_severity = {str(s): float(np.mean([acc[f"{c}/{s}"] for c in CORRUPTIONS])) for s in SEVERITIES}
            mean_corrupted = float(np.mean(list(by_corruption.values())))
            result = {
                "experiment": experiment, "run": run, "model": cfg["model"]["name"],
                "n_images_per_setting": len(subset), "subset_rule": f"{args.per_class} per class, default_rng(0)",
                "clean_acc": clean, "mean_corrupted_acc": mean_corrupted,
                "relative_drop": (clean - mean_corrupted) / clean,
                "by_group": by_group, "by_severity": by_severity, "by_corruption": by_corruption,
                "by_corruption_severity": acc, "eval_minutes": minutes, "hardware": get_hardware_info(),
            }
            out_file.write_text(json.dumps(result, indent=2))
            print(f"{experiment}/{run}: clean {100 * clean:.2f}%, corrupted mean {100 * mean_corrupted:.2f}%, "
                  f"blur {100 * by_group['blur']:.2f}% ({minutes:.1f} min) -> {out_file}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
