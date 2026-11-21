import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.visualize import (BAD, DISPLAY, INK, OK, draw_grid, shared_image_list,
                               upscale)
from src.data import CIFAR10_CLASSES, build_transforms, load_raw_datasets
from src.interpret import attention_rollout_for_vit, grad_cam_for_cnn, overlay_heatmap
from src.models import build_model, get_gradcam_layer, get_model_data_config
from src.utils import load_config

VIT_EXPERIMENTS = ["vit_small_finetune", "small_vit_scratch"]
CNN_EXPERIMENTS = ["resnet50_finetune", "convnext_tiny_finetune", "small_cnn_scratch"]
LABELS = {
    "vit_small_finetune": "ViT-S/16 (DeiT)", "small_vit_scratch": "SmallViT",
    "resnet50_finetune": "ResNet-50", "convnext_tiny_finetune": "ConvNeXt-T",
    "small_cnn_scratch": "SmallCNN",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Attention rollout maps for the trained ViTs")
    parser.add_argument("--run", default="frac1.0_augbasic_seed0", help="run folder name to load")
    parser.add_argument("--outputs-dir", default="outputs",
                        help="folder holding <experiment>/<run>/best.pt (relative to ROOT or absolute)")
    parser.add_argument("--image-list", default="results/gradcam_images.json")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--errors", type=int, default=8, help="misclassified images per model")
    parser.add_argument("--head-fusion", default="mean", choices=["mean", "max", "min"])
    parser.add_argument("--discard-ratio", type=float, default=0.0)
    parser.add_argument("--compare-cnn", action="store_true",
                        help="also draw Grad-CAM of the CNNs next to the ViT rollout (RQ4 figure)")
    parser.add_argument("--out", default="results/figures/attention")
    return parser.parse_args()


def load_run(experiment, run, outputs_dir, device):
    run_dir = outputs_dir / experiment / run
    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        print(f"skip {experiment}: no {checkpoint}")
        return None
    cfg = load_config(run_dir / "config.yaml")
    model_cfg = {**cfg["model"], "pretrained": False}
    model = build_model(model_cfg, len(CIFAR10_CLASSES))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    data_cfg = get_model_data_config(model, cfg["model"], cfg["data"])
    transform = build_transforms(cfg["data"], data_cfg, "none", is_train=False)
    return {"model": model, "transform": transform, "cfg": cfg, "run_dir": run_dir}


def heatmaps_for(entry, raw_images, device, args, batch=16):
    # ViT -> attention rollout, CNN -> Grad-CAM for the predicted class.
    model, transform = entry["model"], entry["transform"]
    is_vit = hasattr(model, "blocks")
    heatmaps, preds = [], []
    for start in range(0, len(raw_images), batch):
        chunk = raw_images[start:start + batch]
        x = torch.stack([transform(Image.fromarray(im)) for im in chunk]).to(device)
        if is_vit:
            maps, logits = attention_rollout_for_vit(model, x, args.head_fusion, args.discard_ratio)
            pred = logits.argmax(1)
        else:
            with torch.no_grad():
                pred = model(x).argmax(1)
            maps, _ = grad_cam_for_cnn(model, get_gradcam_layer(model, entry["cfg"]["model"]), x, pred)
        heatmaps += [np.asarray(Image.fromarray((h * 255).astype(np.uint8)).resize(
            (DISPLAY, DISPLAY), Image.BILINEAR)) / 255.0 for h in maps]
        preds += pred.cpu().tolist()
    return heatmaps, preds


def shared_figures(loaded, indices, test_images, test_labels, device, args, out, prefix, title):
    results = {name: heatmaps_for(entry, test_images[indices], device, args)
               for name, entry in loaded.items()}
    columns = ["image"] + [LABELS[name] for name in loaded]
    half = len(indices) // 2
    for part, chunk in [("a", range(0, half)), ("b", range(half, len(indices)))]:
        rows = []
        for i in chunk:
            idx, true = indices[i], test_labels[indices[i]]
            row = [(upscale(test_images[idx]), f"#{idx} {CIFAR10_CLASSES[true]}", INK)]
            for name in loaded:
                heat, preds = results[name]
                correct = preds[i] == true
                row.append((overlay_heatmap(upscale(test_images[idx]), heat[i]),
                            f"{CIFAR10_CLASSES[preds[i]]} ({'ok' if correct else 'wrong'})",
                            OK if correct else BAD))
            rows.append(row)
        draw_grid(rows, columns, out / f"{prefix}_{part}.png", f"{title} ({part})")


def error_figures(loaded, test_images, test_labels, device, args, out):
    for name, entry in loaded.items():
        npz = entry["run_dir"] / "test_predictions.npz"
        if not npz.exists():
            print(f"skip error figure for {name}: no {npz.name}")
            continue
        pred = np.load(npz)
        wrong = np.flatnonzero(pred["preds"] != pred["targets"])[:args.errors]
        heat, preds = heatmaps_for(entry, test_images[wrong], device, args)
        rows = [[(upscale(test_images[idx]), f"#{idx} true: {CIFAR10_CLASSES[test_labels[idx]]}", INK),
                 (overlay_heatmap(upscale(test_images[idx]), heat[k]),
                  f"predicted: {CIFAR10_CLASSES[preds[k]]}", BAD)]
                for k, idx in enumerate(wrong)]
        draw_grid(rows, ["image", "rollout"], out / f"rollout_errors_{name}.png",
                  f"{LABELS[name]}: first {len(wrong)} misclassified test images")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.is_absolute():
        outputs_dir = ROOT / outputs_dir
    out = ROOT / args.out

    base_cfg = load_config(ROOT / "configs" / "base.yaml")
    base_cfg["data"]["root"] = str(ROOT / base_cfg["data"]["root"])
    _, test_set = load_raw_datasets(base_cfg)
    test_images, test_labels = test_set.data, np.asarray(test_set.targets)
    indices = shared_image_list(args.image_list, test_labels, args.per_class)

    vits = {name: entry for name in VIT_EXPERIMENTS
            if (entry := load_run(name, args.run, outputs_dir, device)) is not None}
    if not vits:
        print("no trained ViT checkpoints found")
        return
    setting = f"{args.head_fusion} heads, discard {args.discard_ratio:g}"

    shared_figures(vits, indices, test_images, test_labels, device, args, out,
                   "rollout_shared", f"Attention rollout · {setting} · shared test images")
    error_figures(vits, test_images, test_labels, device, args, out)

    if args.compare_cnn:
        cnns = {name: entry for name in CNN_EXPERIMENTS
                if (entry := load_run(name, args.run, outputs_dir, device)) is not None}
        # Pretrained models first, then the from-scratch pair, so columns read CNN vs ViT.
        order = ["resnet50_finetune", "convnext_tiny_finetune", "vit_small_finetune",
                 "small_cnn_scratch", "small_vit_scratch"]
        both = {name: (cnns | vits)[name] for name in order if name in (cnns | vits)}
        shared_figures(both, indices, test_images, test_labels, device, args, out,
                       "rq4_gradcam_vs_rollout",
                       "Grad-CAM (CNN) vs attention rollout (ViT) · shared test images")


if __name__ == "__main__":
    main()
