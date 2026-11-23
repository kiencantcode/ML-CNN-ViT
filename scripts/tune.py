import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_experiments import log_line
from src.utils import load_config, run_name

# same search space and same number of trials for every model
LR_SCALES = [0.3, 1.0, 3.0]
WEIGHT_DECAYS = [0.05, 0.005]
DEFAULT = (1.0, 0.05)
FRACTION = 0.1
SEED = 0
CONFIGS = ["small_cnn_scratch", "resnet50_finetune", "convnext_tiny_finetune",
           "resnet50_frozen", "convnext_tiny_frozen", "resnet50_random"]


def trial_overrides(name, scale, weight_decay):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    train = cfg["train"]
    return {"experiment": f"{name}_lrx{scale:g}_wd{weight_decay:g}", "output_dir": "outputs/tuning",
            "seed": SEED, "data.train_fraction": FRACTION, "data.augmentation": "basic",
            "train.lr_head": float(f"{train['lr_head'] * scale:.6g}"),
            "train.lr_backbone": float(f"{train['lr_backbone'] * scale:.6g}"),
            "train.weight_decay": weight_decay, "train.skip_test": True,
            "train.keep_checkpoint": False, "progress_bar": False}


def trials():
    for name in CONFIGS:
        for scale, wd in itertools.product(LR_SCALES, WEIGHT_DECAYS):
            over = trial_overrides(name, scale, wd)
            cfg = load_config(ROOT / "configs" / f"{name}.yaml", over)
            run_dir = ROOT / cfg["output_dir"] / cfg["experiment"] / run_name(cfg)
            yield name, scale, wd, over, run_dir


def summarise():
    rows = {}
    for name, scale, wd, over, run_dir in trials():
        path = run_dir / "metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        rows.setdefault(name, []).append({
            "lr_scale": scale, "weight_decay": wd, "lr_head": over["train.lr_head"],
            "lr_backbone": over["train.lr_backbone"], "val_acc": m["best_val_acc"],
            "best_epoch": m["best_epoch"], "train_time_s": m["train_time_s"]})
    result = {"lr_scales": LR_SCALES, "weight_decays": WEIGHT_DECAYS, "train_fraction": FRACTION,
              "seed": SEED, "rule": "highest validation accuracy, ties go to the default",
              "models": {}}
    lines = ["| Model | LR scale | Weight decay | Val acc (%) | Best epoch | |",
             "|---|---|---|---|---|---|"]
    for name, runs in rows.items():
        best = max(runs, key=lambda r: (r["val_acc"], (r["lr_scale"], r["weight_decay"]) == DEFAULT))
        result["models"][name] = {"selected": best if len(runs) == 6 else None, "trials": runs}
        for r in runs:
            mark = "selected" if r is best and len(runs) == 6 else ""
            lines.append(f"| {name} | {r['lr_scale']:g} | {r['weight_decay']:g} | "
                         f"{100 * r['val_acc']:.2f} | {r['best_epoch']} | {mark} |")
    (ROOT / "results" / "tuning.json").write_text(json.dumps(result, indent=2))
    (ROOT / "results" / "tuning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Same small hyperparameter search for every model")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if not args.summary_only:
        log_file = ROOT / "outputs" / "logs" / "tuning.log"
        todo = [t for t in trials() if not (t[4] / "metrics.json").exists()]
        log_line(log_file, f"start: {len(todo)} trials to do")
        failed = 0
        for i, (name, scale, wd, over, run_dir) in enumerate(todo, 1):
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "-W", "ignore", "scripts/train.py", "--config",
                       f"configs/{name}.yaml", "--skip-existing", "--set",
                       *[f"{k}={v}" for k, v in over.items()]]
            with open(run_dir / "train.log", "w", encoding="utf-8") as out:
                code = subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
            ok = code == 0 and (run_dir / "metrics.json").exists()
            failed += not ok
            log_line(log_file, f"({i}/{len(todo)}) {over['experiment']} {'done' if ok else f'FAILED (exit {code})'}")
        log_line(log_file, f"finished: {len(todo) - failed} ok, {failed} failed")
    summarise()


if __name__ == "__main__":
    main()
