import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CIFAR10_CLASSES, build_transforms, load_raw_datasets
from src.interpret import grad_cam_for_cnn, overlay_heatmap
from src.models import build_model, get_gradcam_layer, get_model_data_config
from src.utils import load_config

EXPERIMENTS = ["resnet50_finetune", "convnext_tiny_finetune", "small_cnn_scratch"]
LABELS = {"resnet50_finetune": "ResNet-50", "convnext_tiny_finetune": "ConvNeXt-Tiny",
          "small_cnn_scratch": "SmallCNN"}
DISPLAY = 128
INK, OK, BAD = "#1f2933", "#0ca30c", "#d03b3b"


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM maps for the trained CNNs")
    parser.add_argument("--run", default="frac1.0_augbasic_seed0", help="run folder name to load")
    parser.add_argument("--image-list", default="results/gradcam_images.json")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--errors", type=int, default=8, help="misclassified images per model")
    parser.add_argument("--out", default="results/figures/gradcam")
    return parser.parse_args()


def shared_image_list(path, test_labels, per_class):
    path = ROOT / path
    if path.exists():
        return json.loads(path.read_text())["indices"]
    rng = np.random.default_rng(0)
    indices = []
    for cls in range(len(CIFAR10_CLASSES)):
        candidates = np.flatnonzero(test_labels == cls)
        indices += sorted(rng.choice(candidates, per_class, replace=False).tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "description": "CIFAR-10 test-set indices used for every Grad-CAM / attention figure",
        "rule": f"{per_class} images per class, numpy default_rng(0), sorted",
        "indices": [int(i) for i in indices],
        "labels": [CIFAR10_CLASSES[test_labels[i]] for i in indices],
    }, indent=2))
    print(f"saved {path}")
    return indices


def load_trained(experiment, run, device):
    run_dir = ROOT / "outputs" / experiment / run
    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        return None
    cfg = load_config(run_dir / "config.yaml")
    model_cfg = {**cfg["model"], "pretrained": False}
    model = build_model(model_cfg, len(CIFAR10_CLASSES))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    data_cfg = get_model_data_config(model, cfg["model"], cfg["data"])
    transform = build_transforms(cfg["data"], data_cfg, "none", is_train=False)
    layer = get_gradcam_layer(model, cfg["model"])
    return model, transform, layer, run_dir


def cams_for(model, transform, layer, raw_images, device, batch=16):
    heatmaps, preds = [], []
    for start in range(0, len(raw_images), batch):
        chunk = raw_images[start:start + batch]
        x = torch.stack([transform(Image.fromarray(im)) for im in chunk]).to(device)
        with torch.no_grad():
            pred = model(x).argmax(1)
        cam, _ = grad_cam_for_cnn(model, layer, x, pred)
        heatmaps += [np.asarray(Image.fromarray((h * 255).astype(np.uint8)).resize(
            (DISPLAY, DISPLAY), Image.BILINEAR)) / 255.0 for h in cam]
        preds += pred.cpu().tolist()
    return heatmaps, preds


def upscale(image):
    return np.asarray(Image.fromarray(image).resize((DISPLAY, DISPLAY), Image.BICUBIC)) / 255.0


def draw_grid(rows, columns, path, title):
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(1.75 * len(columns), 1.9 * len(rows)),
                             squeeze=False)
    for r, row in enumerate(rows):
        for c, (image, caption, color) in enumerate(row):
            ax = axes[r, c]
            ax.imshow(image)
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(caption, fontsize=7.5, color=color, pad=2)
    for c, name in enumerate(columns):
        axes[0, c].annotate(name, (0.5, 1.32), xycoords="axes fraction", ha="center",
                            fontsize=9, fontweight="bold", color=INK)
    fig.suptitle(title, fontsize=11, color=INK, y=1.0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = load_config(ROOT / "configs" / "base.yaml")
    base_cfg["data"]["root"] = str(ROOT / base_cfg["data"]["root"])
    _, test_set = load_raw_datasets(base_cfg)
    test_images, test_labels = test_set.data, np.asarray(test_set.targets)
    indices = shared_image_list(args.image_list, test_labels, args.per_class)

    loaded = {}
    for experiment in EXPERIMENTS:
        result = load_trained(experiment, args.run, device)
        if result is None:
            print(f"skip {experiment}: no checkpoint for {args.run}")
            continue
        loaded[experiment] = result
    if not loaded:
        print("no trained checkpoints found")
        return

    shared = {}
    for experiment, (model, transform, layer, _) in loaded.items():
        shared[experiment] = cams_for(model, transform, layer, test_images[indices], device)

    columns = ["image"] + [LABELS[e] for e in loaded]
    out = ROOT / args.out
    half = len(indices) // 2
    for part, chunk in [("a", range(0, half)), ("b", range(half, len(indices)))]:
        rows = []
        for i in chunk:
            true = test_labels[indices[i]]
            row = [(upscale(test_images[indices[i]]), f"#{indices[i]} {CIFAR10_CLASSES[true]}", INK)]
            for experiment in loaded:
                heat, preds = shared[experiment]
                pred = preds[i]
                mark = "ok" if pred == true else "wrong"
                row.append((overlay_heatmap(upscale(test_images[indices[i]]), heat[i]),
                            f"{CIFAR10_CLASSES[pred]} ({mark})", OK if pred == true else BAD))
            rows.append(row)
        draw_grid(rows, columns, out / f"gradcam_shared_{part}.png",
                  f"Grad-CAM for the predicted class · shared test images ({part})")

    for experiment, (model, transform, layer, run_dir) in loaded.items():
        npz = run_dir / "test_predictions.npz"
        if not npz.exists():
            continue
        pred = np.load(npz)
        wrong = np.flatnonzero(pred["preds"] != pred["targets"])[:args.errors]
        heat, preds = cams_for(model, transform, layer, test_images[wrong], device)
        rows = []
        for k, idx in enumerate(wrong):
            true = test_labels[idx]
            rows.append([
                (upscale(test_images[idx]), f"#{idx} true: {CIFAR10_CLASSES[true]}", INK),
                (overlay_heatmap(upscale(test_images[idx]), heat[k]),
                 f"predicted: {CIFAR10_CLASSES[preds[k]]}", BAD),
            ])
        draw_grid(rows, ["image", "Grad-CAM"], out / f"gradcam_errors_{experiment}.png",
                  f"{LABELS[experiment]}: first {len(wrong)} misclassified test images")


if __name__ == "__main__":
    main()
