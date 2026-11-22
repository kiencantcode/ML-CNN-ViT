"""Per-class test accuracy and most confused class pairs, from the saved test predictions.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")

MAIN_MODELS = {
    "small_cnn_scratch": "SmallCNN",
    "resnet50_finetune": "ResNet-50",
    "convnext_tiny_finetune": "ConvNeXt-T",
    "small_vit_scratch_tuned": "SmallViT",
    "vit_small_finetune_tuned": "DeiT-S",
}
NUM_CLASSES = len(CIFAR10_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Per-class accuracy and confused pairs (3 seeds)")
    parser.add_argument("--outputs", nargs="+", default=[str(OUTPUTS_DIR)],
                        help="one or more folders that hold the runs (first match wins)")
    parser.add_argument("--fraction", default="1.0")
    parser.add_argument("--augmentation", default="basic")
    parser.add_argument("--top-pairs", type=int, default=5)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    return parser.parse_args()


def find_runs(output_dirs, experiment, fraction, augmentation):
    """Return {run_name: npz_path}; a run found in an earlier folder is not replaced."""
    runs = {}
    pattern = f"frac{fraction}_aug{augmentation}_seed*/test_predictions.npz"
    for folder in output_dirs:
        for path in sorted(Path(folder, experiment).glob(pattern)):
            runs.setdefault(path.parent.name, path)
    return runs


def confusion(preds, targets):
    """10x10 matrix: rows = true class, columns = predicted class."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (targets, preds), 1)
    return matrix


def analyse(args):
    per_class, pairs = {}, []
    for experiment, label in MAIN_MODELS.items():
        runs = find_runs(args.outputs, experiment, args.fraction, args.augmentation)
        if not runs:
            print(f"warning: no test_predictions.npz for {experiment}, skipped")
            continue

        matrices = []
        for path in runs.values():
            data = np.load(path)
            matrices.append(confusion(data["preds"].astype(int), data["targets"].astype(int)))
        matrices = np.stack(matrices)                                   # (seeds, 10, 10)

        class_acc = matrices.diagonal(axis1=1, axis2=2) / matrices.sum(axis=2) * 100   # (seeds, 10)
        per_class[label] = (class_acc.mean(axis=0), class_acc.std(axis=0, ddof=1), len(runs))

        # merge both directions (cat->dog and dog->cat), average errors per seed
        total = matrices.sum(axis=0)
        symmetric = total + total.T
        upper = [(symmetric[a, b], a, b) for a in range(NUM_CLASSES) for b in range(a + 1, NUM_CLASSES)]
        for errors, a, b in sorted(upper, reverse=True)[:args.top_pairs]:
            pairs.append({"model": label, "class_a": CIFAR10_CLASSES[a], "class_b": CIFAR10_CLASSES[b],
                          "a_as_b": total[a, b] / len(runs), "b_as_a": total[b, a] / len(runs),
                          "errors_per_seed": errors / len(runs)})
    return per_class, pd.DataFrame(pairs)


def write_tables(per_class, pairs, out_dir):
    rows = []
    for class_index, name in enumerate(CIFAR10_CLASSES):
        row = {"class": name}
        for label, (mean, std, _) in per_class.items():
            row[f"{label}_mean"] = round(mean[class_index], 2)
            row[f"{label}_std"] = round(std[class_index], 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "per_class_accuracy.csv", index=False)
    pairs.round(1).to_csv(out_dir / "confused_pairs.csv", index=False)

    labels = list(per_class)
    seeds = {per_class[label][2] for label in labels}
    lines = [f"Test accuracy per class (%), mean ± std over {'/'.join(map(str, sorted(seeds)))} seeds, "
             "100% data, basic augmentation. 1,000 test images per class (1 image = 0.1 points).", "",
             "| Class | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for class_index, name in enumerate(CIFAR10_CLASSES):
        cells = [f"{per_class[l][0][class_index]:.1f} ± {per_class[l][1][class_index]:.1f}" for l in labels]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Most confused pairs (both directions, errors per seed):", "",
              "| Model | Pair | a → b | b → a | Total |", "|---|---|---|---|---|"]
    for row in pairs.itertuples():
        lines.append(f"| {row.model} | {row.class_a} ↔ {row.class_b} | {row.a_as_b:.1f} "
                     f"| {row.b_as_a:.1f} | {row.errors_per_seed:.1f} |")
    (out_dir / "per_class_accuracy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(per_class, path):
    labels = list(per_class)
    values = np.array([per_class[label][0] for label in labels]).T      # (classes, models)
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2.5, 6))
    image = ax.imshow(values, cmap="viridis", vmin=max(0, values.min() - 5), vmax=100, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] < values.min() + 0.6 * (100 - values.min()) else "black"
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(NUM_CLASSES), CIFAR10_CLASSES)
    ax.set_title("Per-class test accuracy (%), mean over seeds, 100% data")
    fig.colorbar(image, ax=ax, fraction=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class, pairs = analyse(args)
    write_tables(per_class, pairs, out_dir)
    plot(per_class, out_dir / "figures" / "per_class_accuracy.png")
    print((out_dir / "per_class_accuracy.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()