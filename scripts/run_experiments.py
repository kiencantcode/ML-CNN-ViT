import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, parse_overrides, run_name

FRACTIONS = [0.01, 0.1, 1.0]
SEEDS = [0, 1, 2]
AUGMENTATIONS = ["none", "basic", "strong"]
CONFIGS = [
    "configs/small_cnn_scratch.yaml",
    "configs/resnet50_finetune.yaml",
    "configs/convnext_tiny_finetune.yaml",
]
AUG_STUDY_FRACTION = 0.1
ABLATION_CONFIGS = [
    "configs/resnet50_frozen.yaml",
    "configs/convnext_tiny_frozen.yaml",
    "configs/resnet50_random.yaml",
]
ABLATION_FRACTIONS = [0.1, 1.0]


def learning_curve_jobs():
    jobs = []
    for seed in SEEDS:
        for fraction in FRACTIONS:
            for config in CONFIGS:
                jobs.append((config, {"seed": seed, "data.train_fraction": fraction,
                                      "data.augmentation": "basic"}))
    return jobs


def augmentation_jobs():
    jobs = []
    for seed in SEEDS:
        for aug in AUGMENTATIONS:
            if aug == "basic":
                continue
            for config in CONFIGS:
                jobs.append((config, {"seed": seed, "data.train_fraction": AUG_STUDY_FRACTION,
                                      "data.augmentation": aug}))
    return jobs


def ablation_jobs():
    jobs = []
    for seed in SEEDS:
        for fraction in ABLATION_FRACTIONS:
            for config in ABLATION_CONFIGS:
                jobs.append((config, {"seed": seed, "data.train_fraction": fraction,
                                      "data.augmentation": "basic"}))
    return jobs


STUDIES = {
    "learning_curve": learning_curve_jobs,
    "augmentation": augmentation_jobs,
    "ablation": ablation_jobs,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the experiment grid one training run at a time")
    parser.add_argument("--study", nargs="+", default=["learning_curve", "augmentation", "ablation"],
                        choices=list(STUDIES))
    parser.add_argument("--only", nargs="*", default=[],
                        help="keep only jobs whose config path contains one of these words")
    parser.add_argument("--dry-run", action="store_true", help="list the jobs without running them")
    parser.add_argument("--log", default="outputs/logs/run_experiments.log")
    return parser.parse_args()


def run_dir_for(config, overrides):
    cfg = load_config(ROOT / config, overrides)
    return ROOT / cfg["output_dir"] / cfg["experiment"] / run_name(cfg)


def keep_checkpoint(overrides):
    return overrides["data.train_fraction"] == 1.0


def log_line(log_file, text):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {text}"
    print(line, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    args = parse_args()
    jobs = []
    for study in args.study:
        jobs += [(study, c, o) for c, o in STUDIES[study]()]
    if args.only:
        jobs = [j for j in jobs if any(word in j[1] for word in args.only)]

    seen, unique = set(), []
    for job in jobs:
        key = run_dir_for(job[1], job[2])
        if key not in seen:
            seen.add(key)
            unique.append(job)
    jobs = unique

    done = [j for j in jobs if (run_dir_for(j[1], j[2]) / "metrics.json").exists()]
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} jobs: {len(done)} finished, {len(todo)} to run")
    if args.dry_run:
        for study, config, overrides in todo:
            print(f"  {study:<15} {Path(config).stem:<24} {run_dir_for(config, overrides).name}")
        return

    log_file = ROOT / args.log
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_line(log_file, f"start: {len(todo)} runs to do ({len(done)} already finished)")
    failures = []
    for i, (study, config, overrides) in enumerate(todo, 1):
        overrides = dict(overrides)
        overrides["train.keep_checkpoint"] = keep_checkpoint(overrides)
        overrides["progress_bar"] = False
        run_dir = run_dir_for(config, overrides)
        label = f"{Path(config).stem}/{run_dir.name}"
        log_line(log_file, f"({i}/{len(todo)}) {study}: {label}")

        run_log = run_dir / "train.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", "-W", "ignore", str(ROOT / "scripts" / "train.py"),
                   "--config", config, "--skip-existing", "--set"]
        command += [f"{k}={v}" for k, v in overrides.items()]
        start = time.perf_counter()
        with open(run_log, "w", encoding="utf-8") as out:
            result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        minutes = (time.perf_counter() - start) / 60

        last = [line for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()]
        summary = last[-1] if last else ""
        if result.returncode == 0 and (run_dir / "metrics.json").exists():
            log_line(log_file, f"    done in {minutes:.1f} min: {summary}")
        else:
            failures.append(label)
            log_line(log_file, f"    FAILED (exit {result.returncode}) after {minutes:.1f} min: {summary}")

    log_line(log_file, f"finished: {len(todo) - len(failures)} ok, {len(failures)} failed")
    for label in failures:
        log_line(log_file, f"    failed: {label}")


if __name__ == "__main__":
    main()
