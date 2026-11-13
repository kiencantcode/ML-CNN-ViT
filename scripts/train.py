import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CIFAR10_CLASSES, build_dataloaders
from src.engine import autocast, build_optimizer, fit
from src.models import build_model, get_model_data_config
from src.utils import load_config, make_run_dir, parse_overrides, run_name, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train one model on CIFAR-10")
    parser.add_argument("--config", required=True, help="path to an experiment YAML file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. seed=1 data.train_fraction=0.1")
    parser.add_argument("--skip-existing", action="store_true",
                        help="do nothing if this run already has metrics.json")
    parser.add_argument("--overfit-batch", type=int, default=0, metavar="STEPS",
                        help="sanity check: train on one batch for STEPS steps and exit")
    return parser.parse_args()


def overfit_one_batch(model, train_loader, cfg, device, steps):
    model.to(device).train()
    images, targets = next(iter(train_loader))
    images, targets = images.to(device), targets.to(device)
    optimizer = build_optimizer(model, cfg)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, True):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % max(1, steps // 10) == 0:
            acc = (logits.argmax(1) == targets).float().mean().item()
            print(f"step {step:>4}  loss {loss.item():.4f}  batch acc {acc:.2f}")
    print("PASS: loss went close to 0" if loss.item() < 0.05 else "CHECK: loss did not reach 0")


def main():
    args = parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    planned_dir = Path(cfg["output_dir"]) / cfg["experiment"] / run_name(cfg)
    if args.skip_existing and (planned_dir / "metrics.json").exists():
        print(f"skip: {planned_dir} already has metrics.json")
        return

    set_seed(cfg["seed"])
    model = build_model(cfg["model"], num_classes=len(CIFAR10_CLASSES))
    model_data_cfg = get_model_data_config(model, cfg["model"], cfg["data"])
    train_loader, val_loader, test_loader, _ = build_dataloaders(cfg, model_data_cfg)

    if args.overfit_batch:
        overfit_one_batch(model, train_loader, cfg, device, args.overfit_batch)
        return

    run_dir = make_run_dir(cfg)
    print(f"run: {run_dir}")
    print(f"model {cfg['model']['name']} | fraction {cfg['data']['train_fraction']} | "
          f"aug {cfg['data']['augmentation']} | seed {cfg['seed']} | "
          f"train {len(train_loader.dataset)} / val {len(val_loader.dataset)} / "
          f"test {len(test_loader.dataset)} | device {device}")
    fit(model, (train_loader, val_loader, test_loader), cfg, run_dir, device)


if __name__ == "__main__":
    main()
