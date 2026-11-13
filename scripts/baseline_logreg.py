import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import get_split_indices, load_raw_datasets
from src.utils import get_hardware_info, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Logistic regression on raw CIFAR-10 pixels")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fractions", type=float, nargs="+", default=[1.0])
    parser.add_argument("--Cs", type=float, nargs="+", default=[0.0001, 0.0003, 0.001, 0.003])
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--out", default="outputs/baseline/metrics.json")
    return parser.parse_args()


def to_features(images):
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


def fit_and_score(x_train, y_train, x_eval, y_eval, C, max_iter, seed):
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    model.fit(x_train, y_train)
    pred = model.predict(x_eval)
    return model, accuracy_score(y_eval, pred), f1_score(y_eval, pred, average="macro")


def run_one(cfg, x_all, y_all, x_test, y_test, Cs, max_iter):
    train_idx, val_idx = get_split_indices(cfg, y_all)
    scaler = StandardScaler().fit(x_all[train_idx])
    x_train, y_train = scaler.transform(x_all[train_idx]), y_all[train_idx]
    x_val, y_val = scaler.transform(x_all[val_idx]), y_all[val_idx]

    search = []
    best = None
    for C in Cs:
        start = time.perf_counter()
        model, val_acc, val_f1 = fit_and_score(x_train, y_train, x_val, y_val, C, max_iter, cfg["seed"])
        search.append({"C": C, "val_acc": val_acc, "val_f1": val_f1,
                       "fit_time_s": time.perf_counter() - start})
        print(f"  C={C:g}: val acc {val_acc:.4f}", flush=True)
        if best is None or val_acc > best[1]:
            best = (model, val_acc, C)

    model, best_val_acc, best_C = best
    test_pred = model.predict(scaler.transform(x_test))
    return {
        "seed": cfg["seed"],
        "train_fraction": cfg["data"]["train_fraction"],
        "n_train": int(len(train_idx)),
        "best_C": best_C,
        "best_val_acc": best_val_acc,
        "test_acc": accuracy_score(y_test, test_pred),
        "test_f1": f1_score(y_test, test_pred, average="macro"),
        "search": search,
    }


def summarise(runs):
    summary = {}
    for fraction in sorted({r["train_fraction"] for r in runs}):
        accs = [r["test_acc"] for r in runs if r["train_fraction"] == fraction]
        f1s = [r["test_f1"] for r in runs if r["train_fraction"] == fraction]
        summary[str(fraction)] = {
            "test_acc_mean": float(np.mean(accs)), "test_acc_std": float(np.std(accs)),
            "test_f1_mean": float(np.mean(f1s)), "test_f1_std": float(np.std(f1s)),
            "n_seeds": len(accs),
        }
    return summary


def main():
    args = parse_args()
    base_cfg = load_config(args.config)
    train_set, test_set = load_raw_datasets(base_cfg)
    x_all, y_all = to_features(train_set.data), np.asarray(train_set.targets)
    x_test, y_test = to_features(test_set.data), np.asarray(test_set.targets)

    total_start = time.perf_counter()
    runs = []
    for fraction in args.fractions:
        for seed in args.seeds:
            cfg = load_config(args.config, {"seed": seed, "data.train_fraction": fraction})
            print(f"fraction {fraction}, seed {seed}", flush=True)
            result = run_one(cfg, x_all, y_all, x_test, y_test, args.Cs, args.max_iter)
            print(f"  -> best C {result['best_C']:g}, test acc {result['test_acc']:.4f}", flush=True)
            runs.append(result)

    output = {
        "model": "logistic_regression_raw_pixels",
        "features": "flattened 32x32x3 pixels, standardised on the training subset",
        "max_iter": args.max_iter,
        "C_grid": args.Cs,
        "summary": summarise(runs),
        "runs": runs,
        "total_time_s": time.perf_counter() - total_start,
        "hardware": get_hardware_info(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["summary"], indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
