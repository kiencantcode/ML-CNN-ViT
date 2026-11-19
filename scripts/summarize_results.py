import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CIFAR10_CLASSES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BASELINE = "#8a8984"

MODELS = {
    "resnet50_finetune": ("ResNet-50 (pretrained)", "#2a78d6", "o"),
    "convnext_tiny_finetune": ("ConvNeXt-Tiny (pretrained)", "#eb6834", "s"),
    "small_cnn_scratch": ("SmallCNN (scratch)", "#1baf7a", "^"),
    "vit_small_finetune_noclip": ("ViT-S/16 (pretrained)", "#eda100", "D"),
    "small_vit_scratch_noclip": ("SmallViT (scratch)", "#e87ba4", "v"),
    "vit_small_finetune": ("ViT-S/16 + grad clip (ablation)", "#9c6b00", "D"),
    "small_vit_scratch": ("SmallViT + grad clip (ablation)", "#a8436f", "v"),
    "resnet50_frozen": ("ResNet-50 (frozen backbone)", "#008300", "o"),
    "convnext_tiny_frozen": ("ConvNeXt-Tiny (frozen backbone)", "#6250d6", "s"),
    "resnet50_random": ("ResNet-50 (random init)", "#e34948", "o"),
}
EXTRA_COLORS = ["#008300", "#6250d6", "#e34948"]
# ablation-only runs: kept out of the learning-curve and augmentation plots
ABLATION_ONLY = {"resnet50_random", "vit_small_finetune", "small_vit_scratch"}
AUG_ORDER = ["none", "basic", "strong"]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect metrics.json files into tables and plots")
    parser.add_argument("--outputs", default="outputs", help="folder that holds the runs")
    parser.add_argument("--results", default="results", help="folder for tables and figures")
    return parser.parse_args()


def style(name):
    if name in MODELS:
        return MODELS[name]
    index = sorted(set(MODELS) | {name}).index(name) % len(EXTRA_COLORS)
    return name, EXTRA_COLORS[index], "P"


def setup_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {path}")


def load_runs(outputs):
    rows = []
    for path in sorted(Path(outputs).glob("*/*/metrics.json")):
        if path.parts[-3].startswith("_") or path.parts[-3] == "baseline":
            continue
        m = json.loads(path.read_text())
        rows.append({
            "experiment": m["experiment"], "model": m["model"],
            "pretrained": m["pretrained"], "freeze_backbone": m["freeze_backbone"],
            "train_fraction": m["train_fraction"], "augmentation": m["augmentation"],
            "seed": m["seed"], "n_train": m["n_train"],
            "test_acc": 100 * m["test_acc"], "test_f1": 100 * m["test_f1"],
            "best_val_acc": 100 * m["best_val_acc"], "best_epoch": m["best_epoch"],
            "epochs": m["epochs"], "params_m": m["params_total"] / 1e6,
            "trainable_m": m["params_trainable"] / 1e6,
            "train_time_min": m["train_time_s"] / 60,
            "images_per_s": m["train_images_per_s"], "peak_gpu_gb": m["peak_gpu_mem_gb"],
            "run_dir": str(path.parent),
        })
    return pd.DataFrame(rows)


def load_baseline(outputs):
    path = Path(outputs) / "baseline" / "metrics.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())["summary"]
    return pd.DataFrame([
        {"train_fraction": float(f), "test_acc": 100 * s["test_acc_mean"],
         "test_acc_std": 100 * s["test_acc_std"]}
        for f, s in summary.items()
    ]).sort_values("train_fraction")


def summarise(runs):
    keys = ["experiment", "model", "pretrained", "freeze_backbone", "train_fraction", "augmentation"]
    grouped = runs.groupby(keys)
    table = grouped.agg(
        n_seeds=("seed", "nunique"),
        test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
        test_f1_mean=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
        best_epoch_mean=("best_epoch", "mean"),
        train_time_min=("train_time_min", "mean"), images_per_s=("images_per_s", "mean"),
        peak_gpu_gb=("peak_gpu_gb", "max"), params_m=("params_m", "first"),
        trainable_m=("trainable_m", "first"),
    ).reset_index()
    return table.fillna({"test_acc_std": 0.0, "test_f1_std": 0.0}).round(3)


def write_markdown(table, path):
    lines = ["| Model | Fraction | Aug | Seeds | Test acc (%) | Macro-F1 (%) | Params (M) | Train time (min) | Peak GPU (GB) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.sort_values(["experiment", "augmentation", "train_fraction"]).iterrows():
        label = style(r["experiment"])[0] + (" [frozen]" if r["freeze_backbone"] else "")
        spread = r["n_seeds"] > 1
        acc = f"{r['test_acc_mean']:.2f}" + (f" ± {r['test_acc_std']:.2f}" if spread else "")
        f1 = f"{r['test_f1_mean']:.2f}" + (f" ± {r['test_f1_std']:.2f}" if spread else "")
        lines.append(
            f"| {label} | {r['train_fraction']:g} | {r['augmentation']} | {r['n_seeds']} | "
            f"{acc} | {f1} | {r['params_m']:.2f} | {r['train_time_min']:.1f} | "
            f"{r['peak_gpu_gb']:.2f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {path}")


def plot_training_curves(runs, fig_dir):
    for _, run in runs.iterrows():
        history = pd.read_csv(Path(run["run_dir"]) / "history.csv")
        name, color, _ = style(run["experiment"])
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, metric, label in [(ax_loss, "loss", "Loss"), (ax_acc, "acc", "Accuracy (%)")]:
            scale = 100 if metric == "acc" else 1
            ax.plot(history["epoch"], history[f"train_{metric}"] * scale, color=color, lw=2,
                    ls="--", marker="o", ms=4, label="train")
            ax.plot(history["epoch"], history[f"val_{metric}"] * scale, color=color, lw=2,
                    marker="s", ms=4, label="validation")
            setup_axes(ax, label, "epoch")
            ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
        best = history.loc[history["val_acc"].idxmax()]
        ax_acc.annotate(f"best val {100 * best['val_acc']:.2f}% (epoch {int(best['epoch'])})",
                        (best["epoch"], 100 * best["val_acc"]), xytext=(0, -28),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
        fig.suptitle(f"{name} · fraction {run['train_fraction']:g} · aug {run['augmentation']} · "
                     f"seed {run['seed']} · test {run['test_acc']:.2f}%",
                     x=0.01, ha="left", fontsize=10, color=INK_2)
        run_name = Path(run["run_dir"]).name
        save(fig, fig_dir / "training_curves" / f"{run['experiment']}_{run_name}.png")


def plot_learning_curve(table, baseline, fig_dir, augmentation="basic"):
    data = table[(table["augmentation"] == augmentation) & (~table["freeze_backbone"])
                 & (~table["experiment"].isin(ABLATION_ONLY))]
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ends = []
    for experiment, group in data.groupby("experiment"):
        group = group.sort_values("train_fraction")
        name, color, marker = style(experiment)
        x, y, s = group["train_fraction"] * 100, group["test_acc_mean"], group["test_acc_std"]
        ax.plot(x, y, color=color, lw=2, marker=marker, ms=7, label=name,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.15, lw=0)
        ends.append([x.iloc[-1], y.iloc[-1], y.iloc[-1]])
    # spread the end labels so close values don't overlap
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        ends[i][2] = max(ends[i][2], ends[i - 1][2] + 3.5)
    for x_end, y_end, y_text in ends:
        ax.annotate(f"{y_end:.1f}", (x_end, y_end), xytext=(x_end * 1.12, y_text),
                    textcoords="data", va="center", fontsize=8.5, color=INK)
    if baseline is not None:
        baseline = baseline[baseline["train_fraction"].isin(set(data["train_fraction"]))]
        ax.plot(baseline["train_fraction"] * 100, baseline["test_acc"], color=BASELINE, lw=1.5,
                ls=":", marker="x", ms=6, label="Logistic regression (baseline)")
    ax.axhline(10, color=GRID, lw=1, ls="--")
    ax.text(1, 11, "random guess 10%", fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ticks = sorted(set((data["train_fraction"] * 100).round(6)))
    ax.set_xticks(ticks, [f"{t:g}%" for t in ticks])
    ax.minorticks_off()
    setup_axes(ax, f"Learning curve (augmentation: {augmentation})",
               "training data used (log scale)", "test accuracy (%)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    save(fig, fig_dir / f"learning_curve_{augmentation}.png")


def plot_model_comparison(table, baseline, fig_dir, fraction=1.0, augmentation="basic"):
    data = table[(table["train_fraction"] == fraction) & (table["augmentation"] == augmentation)]
    data = data.sort_values("test_acc_mean")
    if data.empty:
        return
    labels, values, errors, colors = [], [], [], []
    for _, r in data.iterrows():
        name, color, _ = style(r["experiment"])
        labels.append(name + (" [frozen]" if r["freeze_backbone"] else "") + f"  (n={r['n_seeds']})")
        values.append(r["test_acc_mean"]), errors.append(r["test_acc_std"]), colors.append(color)
    if baseline is not None and fraction in set(baseline["train_fraction"]):
        row = baseline[baseline["train_fraction"] == fraction].iloc[0]
        labels.insert(0, "Logistic regression (baseline)")
        values.insert(0, row["test_acc"]), errors.insert(0, row["test_acc_std"])
        colors.insert(0, BASELINE)

    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(labels) + 1.4))
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.6, color=colors, xerr=errors,
            error_kw=dict(ecolor=INK_2, lw=1, capsize=3))
    for yi, v, e, label in zip(y, values, errors, labels):
        text = f"{v:.2f}" if "(n=1)" in label else f"{v:.2f} ± {e:.2f}"
        ax.text(v + e + 1.2, yi, text, va="center", fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    setup_axes(ax, f"Test accuracy · {fraction * 100:g}% data · aug {augmentation}",
               "test accuracy (%)")
    ax.grid(axis="y", visible=False)
    save(fig, fig_dir / f"model_comparison_frac{fraction:g}_{augmentation}.png")


def plot_cost(table, fig_dir, fraction=1.0, augmentation="basic"):
    data = table[(table["train_fraction"] == fraction) & (table["augmentation"] == augmentation)]
    if data.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    data = data.sort_values("test_acc_mean", ascending=False)
    for ax, column, xlabel in [(axes[0], "train_time_min", "training time (min)"),
                               (axes[1], "trainable_m", "trainable parameters (millions, log scale)")]:
        for _, r in data.iterrows():
            name, color, marker = style(r["experiment"])
            ax.errorbar(r[column], r["test_acc_mean"], yerr=r["test_acc_std"], fmt=marker,
                        color=color, ms=9, mec=SURFACE, mew=1.5, ecolor=INK_2, capsize=3,
                        label=name)
        setup_axes(ax, xlabel=xlabel)
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].set_ylabel("test accuracy (%)", color=INK_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.27)
    fig.suptitle(f"Accuracy vs compute cost · {fraction * 100:g}% data · aug {augmentation}",
                 x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    save(fig, fig_dir / f"accuracy_vs_cost_frac{fraction:g}_{augmentation}.png")


def plot_augmentation(table, fig_dir):
    data = table[~table["freeze_backbone"] & (~table["experiment"].isin(ABLATION_ONLY))]
    if data["augmentation"].nunique() < 2:
        return
    for fraction, frac_data in data.groupby("train_fraction"):
        if frac_data["augmentation"].nunique() < 2:
            continue
        experiments = sorted(frac_data["experiment"].unique())
        fig, ax = plt.subplots(figsize=(7.5, 4))
        width = 0.8 / len(AUG_ORDER)
        hatches = {"none": "", "basic": "///", "strong": "xxx"}
        for i, aug in enumerate(AUG_ORDER):
            for j, experiment in enumerate(experiments):
                row = frac_data[(frac_data["experiment"] == experiment) & (frac_data["augmentation"] == aug)]
                if row.empty:
                    continue
                r = row.iloc[0]
                color = style(experiment)[1]
                x = j + (i - 1) * width
                ax.bar(x, r["test_acc_mean"], width * 0.92, color=color, hatch=hatches[aug],
                       edgecolor=SURFACE, yerr=r["test_acc_std"],
                       error_kw=dict(ecolor=INK_2, lw=1, capsize=2),
                       label=aug if j == 0 else None)
                ax.text(x, r["test_acc_mean"] + r["test_acc_std"] + 0.5, f"{r['test_acc_mean']:.1f}",
                        ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(experiments)),
                      [style(e)[0].replace(" (", "\n(") for e in experiments], fontsize=8.5)
        ax.set_ylim(0, 108)
        setup_axes(ax, f"Augmentation study · {fraction * 100:g}% data",
                   ylabel="test accuracy (%)")
        legend = ax.legend(title="augmentation", frameon=False, fontsize=8.5, ncol=3,
                           loc="upper center", bbox_to_anchor=(0.5, -0.16))
        for patch in legend.get_patches():
            patch.set_facecolor(BASELINE)
        save(fig, fig_dir / f"augmentation_frac{fraction:g}.png")


def plot_confusion_matrices(runs, fig_dir, fraction=1.0, augmentation="basic"):
    data = runs[(runs["train_fraction"] == fraction) & (runs["augmentation"] == augmentation)
                & (~runs["freeze_backbone"])]
    for experiment, group in data.groupby("experiment"):
        run = group.sort_values("seed").iloc[0]
        npz = Path(run["run_dir"]) / "test_predictions.npz"
        if not npz.exists():
            continue
        pred = np.load(npz)
        n = len(CIFAR10_CLASSES)
        matrix = np.zeros((n, n), dtype=int)
        np.add.at(matrix, (pred["targets"], pred["preds"]), 1)
        percent = 100 * matrix / matrix.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        image = ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
        for i in range(n):
            for j in range(n):
                if percent[i, j] >= 1:
                    ax.text(j, i, f"{percent[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                            color="white" if percent[i, j] > 60 else INK)
        ax.set_xticks(range(n), CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(range(n), CIFAR10_CLASSES, fontsize=8.5)
        ax.set_xlabel("predicted class", color=INK_2, fontsize=9)
        ax.set_ylabel("true class", color=INK_2, fontsize=9)
        name = style(experiment)[0]
        ax.set_title(f"{name} · seed {run['seed']} · test {run['test_acc']:.2f}%\n"
                     "row-normalised % (cells under 1% hidden)", loc="left", fontsize=10, color=INK)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("% of true class", fontsize=8.5)
        save(fig, fig_dir / f"confusion_{experiment}_frac{fraction:g}_{augmentation}.png")


def summarise_cifar10c(results, fig_dir):
    files = sorted((results / "cifar10c").glob("*.json"))
    if not files:
        return
    data = [json.loads(f.read_text()) for f in files]
    rows = []
    for d in data:
        row = {"experiment": d["experiment"], "run": d["run"], "clean_acc": 100 * d["clean_acc"],
               "mean_corrupted_acc": 100 * d["mean_corrupted_acc"],
               "relative_drop_pct": 100 * d["relative_drop"]}
        row.update({f"{g}_acc": 100 * v for g, v in d["by_group"].items()})
        row.update({f"severity{s}_acc": 100 * v for s, v in d["by_severity"].items()})
        row.update({f"{c}_acc": 100 * v for c, v in d["by_corruption"].items()})
        rows.append(row)
    table = pd.DataFrame(rows).round(2)
    table.to_csv(results / "cifar10c_summary.csv", index=False)
    lines = ["| Model | Run | Clean | Corrupted mean | Drop (rel. %) | Noise | Blur | Weather | Digital |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {style(r['experiment'])[0]} | {r['run']} | {r['clean_acc']:.2f} | "
                     f"{r['mean_corrupted_acc']:.2f} | {r['relative_drop_pct']:.1f} | {r['noise_acc']:.2f} | "
                     f"{r['blur_acc']:.2f} | {r['weather_acc']:.2f} | {r['digital_acc']:.2f} |")
    (results / "cifar10c_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {results / 'cifar10c_summary.md'}")

    by_model = {}
    for d in data:
        by_model.setdefault(d["experiment"], []).append(d)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for experiment, runs in by_model.items():
        name, color, marker = style(experiment)
        ys = np.array([[100 * r["clean_acc"]] + [100 * r["by_severity"][str(s)] for s in range(1, 6)]
                       for r in runs])
        mean = ys.mean(axis=0)
        ax.plot(range(6), mean, color=color, lw=2, marker=marker, ms=7, mec=SURFACE, mew=1.5,
                label=f"{name} (n={len(runs)})")
        ax.annotate(f"{mean[-1]:.1f}", (5, mean[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(6), ["clean", "1", "2", "3", "4", "5"])
    setup_axes(ax, "CIFAR-10-C: accuracy vs corruption severity (mean of 19 corruptions)",
               "severity", "accuracy (%)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2)
    save(fig, fig_dir / "cifar10c_severity.png")

    groups = ["noise", "blur", "weather", "digital"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    experiments = list(by_model)
    width = 0.8 / max(1, len(experiments))
    for j, experiment in enumerate(experiments):
        name, color, _ = style(experiment)
        values = [np.mean([100 * r["by_group"][g] for r in by_model[experiment]]) for g in groups]
        x = np.arange(len(groups)) + (j - (len(experiments) - 1) / 2) * width
        ax.bar(x, values, width * 0.92, color=color, edgecolor=SURFACE, label=name)
        for xi, v in zip(x, values):
            ax.text(xi, v + 0.8, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(groups)), groups)
    setup_axes(ax, "CIFAR-10-C: accuracy per corruption group (mean of severities 1-5)",
               ylabel="accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper center",
              bbox_to_anchor=(0.5, -0.1), ncol=len(experiments))
    save(fig, fig_dir / "cifar10c_groups.png")

    blur = ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"]
    fig, axes = plt.subplots(1, len(blur), figsize=(13, 3.2), sharey=True)
    for ax, corruption in zip(axes, blur):
        for experiment, runs in by_model.items():
            name, color, marker = style(experiment)
            ys = np.mean([[100 * r["clean_acc"]] + [100 * r["by_corruption_severity"][f"{corruption}/{s}"]
                                                     for s in range(1, 6)] for r in runs], axis=0)
            ax.plot(range(6), ys, color=color, lw=2, marker=marker, ms=5, label=name)
        ax.set_xticks(range(6), ["c", "1", "2", "3", "4", "5"])
        setup_axes(ax, corruption.replace("_", " "), "severity")
    axes[0].set_ylabel("accuracy (%)", color=INK_2, fontsize=9)
    fig.suptitle("CIFAR-10-C blur corruptions (c = clean)", x=0.01, ha="left", fontsize=11,
                 color=INK, fontweight="bold")
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(labels))
    save(fig, fig_dir / "cifar10c_blur.png")


def plot_ablation(table, fig_dir):
    pairs = [("resnet50_finetune", "ResNet-50 full fine-tune"),
             ("resnet50_frozen", "ResNet-50 frozen backbone"),
             ("resnet50_random", "ResNet-50 random init"),
             ("convnext_tiny_finetune", "ConvNeXt-Tiny full fine-tune"),
             ("convnext_tiny_frozen", "ConvNeXt-Tiny frozen backbone")]
    data = table[(table["augmentation"] == "basic") & table["experiment"].isin([p[0] for p in pairs])]
    fractions = sorted(set(data["train_fraction"]) & {0.1, 1.0})
    if data["experiment"].nunique() < 2 or not fractions:
        return
    fig, axes = plt.subplots(1, len(fractions), figsize=(6 * len(fractions), 3.6), squeeze=False)
    for ax, fraction in zip(axes[0], fractions):
        labels, values, errors, colors = [], [], [], []
        for experiment, label in pairs:
            row = data[(data["experiment"] == experiment) & (data["train_fraction"] == fraction)]
            if row.empty:
                continue
            r = row.iloc[0]
            labels.append(f"{label} (n={r['n_seeds']})")
            values.append(r["test_acc_mean"]), errors.append(r["test_acc_std"])
            colors.append(style(experiment)[1])
        y = np.arange(len(labels))[::-1]
        ax.barh(y, values, height=0.6, color=colors, xerr=errors,
                error_kw=dict(ecolor=INK_2, lw=1, capsize=3))
        for yi, v, e in zip(y, values, errors):
            ax.text(v + e + 1, yi, f"{v:.1f}", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y, labels, fontsize=8.5)
        ax.set_xlim(0, 112)
        setup_axes(ax, f"{fraction * 100:g}% data", "test accuracy (%)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Ablation: pretraining and fine-tuning", x=0.01, ha="left", fontsize=11,
                 color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, fig_dir / "ablation.png")


def main():
    args = parse_args()
    results = Path(args.results)
    fig_dir = results / "figures"
    runs = load_runs(args.outputs)
    baseline = load_baseline(args.outputs)
    if runs.empty:
        print(f"no finished runs found in {args.outputs}")
        return

    results.mkdir(parents=True, exist_ok=True)
    runs.drop(columns=["run_dir"]).to_csv(results / "all_runs.csv", index=False)
    table = summarise(runs)
    table.to_csv(results / "summary.csv", index=False)
    write_markdown(table, results / "summary.md")
    print(f"{len(runs)} runs, {len(table)} settings")
    print(table[["experiment", "train_fraction", "augmentation", "n_seeds",
                 "test_acc_mean", "test_acc_std"]].to_string(index=False))

    plot_training_curves(runs, fig_dir)
    for aug in sorted(table["augmentation"].unique()):
        plot_learning_curve(table, baseline, fig_dir, aug)
    plot_model_comparison(table, baseline, fig_dir)
    plot_cost(table, fig_dir)
    plot_augmentation(table, fig_dir)
    plot_ablation(table, fig_dir)
    plot_confusion_matrices(runs, fig_dir)
    summarise_cifar10c(results, fig_dir)


if __name__ == "__main__":
    main()
