import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.data import release_workers
from src.utils import count_parameters, get_hardware_info


def optimizer_steps_per_epoch(loader, grad_accum_steps):
    return math.ceil(len(loader) / grad_accum_steps)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    train_cfg = cfg["train"]
    total_steps = max(1, train_cfg["epochs"] * steps_per_epoch)
    warmup_steps = min(int(train_cfg.get("warmup_epochs", 0) * steps_per_epoch), total_steps - 1)
    min_lr_ratio = train_cfg.get("min_lr_ratio", 0.0)

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def set_train_mode(model, freeze_backbone):
    if freeze_backbone:
        model.eval()
        model.get_classifier().train()
    else:
        model.train()


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=enabled and device.type == "cuda")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, cfg):
    train_cfg = cfg["train"]
    accum = train_cfg.get("grad_accum_steps", 1)
    amp = train_cfg.get("amp", True)
    grad_clip = train_cfg.get("grad_clip")  # optional, e.g. 1.0 for ViT; off when unset
    set_train_mode(model, cfg["model"].get("freeze_backbone", False))

    total_loss, correct, seen = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc="train", leave=False, disable=not cfg.get("progress_bar", True))
    for step, (images, targets) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device, amp):
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became {loss.item()} at step {step}; lower the learning rate")

        scaler.scale(loss / accum).backward()
        is_last = step + 1 == len(loader)
        if (step + 1) % accum == 0 or is_last:
            if grad_clip:
                scaler.unscale_(optimizer)  # clip the real gradients, not the scaled ones
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        batch = targets.size(0)
        total_loss += loss.item() * batch
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch
        progress.set_postfix(loss=f"{total_loss / seen:.3f}", acc=f"{correct / seen:.3f}")

    return {"loss": total_loss / seen, "acc": correct / seen, "lr": optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp=True, return_predictions=False):
    model.eval()
    total_loss, seen = 0.0, 0
    all_preds, all_targets = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        seen += targets.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_targets.append(targets.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    result = {
        "loss": total_loss / seen,
        "acc": float((preds == targets).mean()),
        "f1": float(f1_score(targets, preds, average="macro")),
    }
    if return_predictions:
        result["preds"], result["targets"] = preds, targets
    return result


def build_optimizer(model, cfg):
    from src.models import get_param_groups

    train_cfg = cfg["train"]
    groups = get_param_groups(model, train_cfg["lr_head"], train_cfg["lr_backbone"],
                              train_cfg["weight_decay"])
    if train_cfg.get("optimizer", "adamw") != "adamw":
        raise ValueError(f"Unsupported optimizer: {train_cfg['optimizer']}")
    return torch.optim.AdamW(groups)


def _rng_state():
    """Random states of every generator training uses, so a resumed run shuffles the same way."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _truncate_history(path, last_epoch):
    """Drop history rows after `last_epoch` (an epoch logged but not checkpointed)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], [r for r in rows[1:] if int(r[0]) <= last_epoch]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([header] + body)


def fit(model, loaders, cfg, run_dir, device, resume=False):
    """Train, keep the best epoch by val accuracy, test it once, write metrics.json.

    Optional features, all off unless the config turns them on, so runs that do not
    set them behave exactly as before:
      train.save_last: true            write last.pt after every epoch (weights,
                                       optimizer, scheduler, scaler, counters, RNG)
      train.early_stopping_patience: N stop after N epochs without val-acc improvement
      train.grad_clip: x               clip the gradient norm to x
    resume=True continues from run_dir/last.pt (needs save_last). Use the same config
    as the interrupted run; changing epochs shifts the LR schedule.
    """
    train_loader, val_loader, test_loader = loaders
    train_cfg = cfg["train"]
    amp = train_cfg.get("amp", True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.get("label_smoothing", 0.0))
    optimizer = build_optimizer(model, cfg)
    steps = optimizer_steps_per_epoch(train_loader, train_cfg.get("grad_accum_steps", 1))
    scheduler = build_scheduler(optimizer, cfg, steps)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    run_dir = Path(run_dir)
    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    save_last = bool(train_cfg.get("save_last", False))
    patience = int(train_cfg.get("early_stopping_patience") or 0)  # 0 = off
    fields = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "epoch_time_s"]
    best_val_acc, best_epoch, epoch_times = -1.0, -1, []
    epochs_without_improvement, epochs_run, early_stopped = 0, 0, False
    elapsed_before, start_epoch = 0.0, 1

    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"cannot resume: {last_path} does not exist "
                                    "(was train.save_last on?)")
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best_val_acc, best_epoch = ckpt["best_val_acc"], ckpt["best_epoch"]
        epoch_times, elapsed_before = ckpt["epoch_times"], ckpt["elapsed"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        early_stopped = ckpt["early_stopped"]
        epochs_run = ckpt["epoch"]
        start_epoch = ckpt["epoch"] + 1
        _set_rng_state(ckpt["rng"])
        _truncate_history(history_path, ckpt["epoch"])
        print(f"resumed from {last_path} at epoch {ckpt['epoch']} "
              f"(best epoch {best_epoch}, val acc {best_val_acc:.4f})", flush=True)
    else:
        with open(history_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    total_start = time.perf_counter()
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        if early_stopped:  # the interrupted run had already stopped early
            break
        start = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                                      scaler, device, cfg)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start
        epoch_times.append(epoch_time)
        val_stats = evaluate(model, val_loader, criterion, device, amp)

        row = [epoch, f"{train_stats['lr']:.3e}", f"{train_stats['loss']:.4f}",
               f"{train_stats['acc']:.4f}", f"{val_stats['loss']:.4f}", f"{val_stats['acc']:.4f}",
               f"{val_stats['f1']:.4f}", f"{epoch_time:.1f}"]
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        epochs_run = epoch
        improved = val_stats["acc"] > best_val_acc
        if improved:
            best_val_acc, best_epoch = val_stats["acc"], epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        if patience and epochs_without_improvement >= patience:
            early_stopped = True

        if save_last:
            # Temp file first, so a crash mid-save cannot corrupt the previous last.pt.
            tmp_path = last_path.with_suffix(".pt.tmp")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "early_stopped": early_stopped,
                "epoch_times": epoch_times,
                "elapsed": elapsed_before + time.perf_counter() - total_start,
                "rng": _rng_state(),
            }, tmp_path)
            tmp_path.replace(last_path)

        print(f"epoch {epoch:>3}/{train_cfg['epochs']}  train loss {train_stats['loss']:.4f} "
              f"acc {train_stats['acc']:.4f}  val loss {val_stats['loss']:.4f} "
              f"acc {val_stats['acc']:.4f}  {epoch_time:.0f}s{'  *best' if improved else ''}",
              flush=True)
        if early_stopped:
            print(f"early stopping: no val acc improvement for {patience} epochs", flush=True)
            break

    train_time = elapsed_before + time.perf_counter() - total_start
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0

    release_workers(train_loader)
    release_workers(val_loader)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_stats = evaluate(model, test_loader, criterion, device, amp, return_predictions=True)
    np.savez_compressed(run_dir / "test_predictions.npz",
                        preds=test_stats.pop("preds"), targets=test_stats.pop("targets"))

    total_params, trainable_params = count_parameters(model)
    n_train = len(train_loader.dataset)
    steady_times = epoch_times[1:] or epoch_times
    metrics = {
        "experiment": cfg["experiment"],
        "model": cfg["model"]["name"],
        "pretrained": cfg["model"].get("pretrained", False),
        "freeze_backbone": cfg["model"].get("freeze_backbone", False),
        "train_fraction": cfg["data"]["train_fraction"],
        "augmentation": cfg["data"]["augmentation"],
        "seed": cfg["seed"],
        "n_train": n_train,
        "test_acc": test_stats["acc"],
        "test_f1": test_stats["f1"],
        "test_loss": test_stats["loss"],
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epochs": train_cfg["epochs"],
        "epochs_run": epochs_run,
        "early_stopped": early_stopped,
        "early_stopping_patience": patience or None,
        "params_total": total_params,
        "params_trainable": trainable_params,
        "train_time_s": train_time,
        "first_epoch_time_s": epoch_times[0],
        "epoch_time_mean_s": float(np.mean(steady_times)),
        "train_images_per_s": n_train * len(steady_times) / sum(steady_times),
        "peak_gpu_mem_gb": peak_mem,
        "hardware": get_hardware_info(),
        "config": cfg,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if not train_cfg.get("keep_checkpoint", True):
        checkpoint_path.unlink(missing_ok=True)
        last_path.unlink(missing_ok=True)

    print(f"test acc {test_stats['acc']:.4f}  macro-F1 {test_stats['f1']:.4f}  "
          f"(best epoch {best_epoch}, {train_time / 60:.1f} min, peak GPU {peak_mem:.2f} GB)")
    return metrics
