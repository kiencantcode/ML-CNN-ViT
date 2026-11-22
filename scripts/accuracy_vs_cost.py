"""RQ3: join test accuracy (3 seeds) with compute cost measured on one GPU (CNN) and (ViT);
cost comes from results/compute_cost.csv (scripts/benchmark_cost.py, one NVIDIA L4).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# main model per family; the order sets the table rows
MAIN_MODELS = {
    "small_cnn_scratch": "SmallCNN (scratch)",
    "resnet50_finetune": "ResNet-50 (pretrained)",
    "convnext_tiny_finetune": "ConvNeXt-Tiny (pretrained)",
    "small_vit_scratch_tuned": "SmallViT (scratch)",
    "vit_small_finetune_tuned": "DeiT-S (pretrained)",
}
# a tuned LR does not change the architecture, so the tuned runs reuse the benchmark of the old run
COST_NAME = {
    "small_vit_scratch_noclip": "small_vit_scratch_tuned",
    "vit_small_finetune_noclip": "vit_small_finetune_tuned",
}
COLORS = {
    "small_cnn_scratch": "#1baf7a",
    "resnet50_finetune": "#2a78d6",
    "convnext_tiny_finetune": "#eb6834",
    "small_vit_scratch_tuned": "#e87ba4",
    "vit_small_finetune_tuned": "#eda100",
}
MARKERS = {name: ("o" if "vit" not in name else "D") for name in MAIN_MODELS}


def parse_args():
    parser = argparse.ArgumentParser(description="Accuracy vs compute cost table and figure (RQ3)")
    parser.add_argument("--cost", default=str(RESULTS_DIR / "compute_cost.csv"))
    parser.add_argument("--summaries", nargs="+",
                        default=[str(RESULTS_DIR / "summary.csv"), str(RESULTS_DIR / "vit" / "summary.csv")])
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--augmentation", default="basic")
    parser.add_argument("--out", default=str(RESULTS_DIR))
    return parser.parse_args()


def load_table(args):
    cost = pd.read_csv(args.cost)
    cost["experiment"] = cost["experiment"].replace(COST_NAME)

    acc = pd.concat([pd.read_csv(path) for path in args.summaries], ignore_index=True)
    acc = acc[(acc["train_fraction"] == args.fraction) & (acc["augmentation"] == args.augmentation)]
    acc = acc[["experiment", "n_seeds", "test_acc_mean", "test_acc_std"]].drop_duplicates("experiment")

    table = cost.merge(acc, on="experiment", how="inner")
    table = table[table["experiment"].isin(MAIN_MODELS)]
    missing = set(MAIN_MODELS) - set(table["experiment"])
    if missing:
        print(f"warning: no cost or accuracy for {sorted(missing)}")

    table["label"] = table["experiment"].map(MAIN_MODELS)
    table["order"] = table["experiment"].map(list(MAIN_MODELS).index)
    return table.sort_values("order").reset_index(drop=True)


def write_tables(table, out_dir):
    columns = ["experiment", "label", "image_size", "params_m", "gflops", "infer_img_s", "train_img_s",
               "epoch_min_100pct", "peak_train_gb", "n_seeds", "test_acc_mean", "test_acc_std", "gpu"]
    table[columns].to_csv(out_dir / "rq3_accuracy_cost.csv", index=False)

    gpu = table["gpu"].iloc[0]
    lines = [
        f"Accuracy: test set, mean ± std over seeds, 100% data, basic augmentation. Cost: one {gpu}.",
        "",
        "| Model | Input | Params (M) | GFLOPs | Inference (img/s) | Train (img/s) "
        "| Epoch at 100% (min) | Peak train GPU (GB) | Test acc (%) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.label} | {row.image_size} | {row.params_m:.2f} | {row.gflops:.2f} "
            f"| {row.infer_img_s:,.0f} | {row.train_img_s:,.0f} | {row.epoch_min_100pct:.1f} "
            f"| {row.peak_train_gb:.2f} | {row.test_acc_mean:.2f} ± {row.test_acc_std:.2f} |"
        )
    (out_dir / "rq3_accuracy_cost.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(table, path):
    panels = [("gflops", "GFLOPs per image (log scale)"),
              ("infer_img_s", "Inference throughput, img/s (log scale)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (column, xlabel) in zip(axes, panels):
        for row in table.itertuples():
            ax.errorbar(getattr(row, column), row.test_acc_mean, yerr=row.test_acc_std,
                        marker=MARKERS[row.experiment], color=COLORS[row.experiment],
                        markersize=9, capsize=3, linestyle="none",
                        label=f"{row.label}, {row.image_size}px")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="major", alpha=0.3)
    axes[0].set_ylabel("Test accuracy (%)")
    # labels overlap where the three 224px models sit together, so one shared legend is used
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Accuracy vs compute cost (100% data, basic augmentation)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = load_table(args)
    write_tables(table, out_dir)
    plot(table, out_dir / "figures" / "rq3_accuracy_vs_cost.png")
    print((out_dir / "rq3_accuracy_cost.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()