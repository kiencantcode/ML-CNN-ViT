import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from src.engine import autocast, build_optimizer
from src.models import build_model
from src.utils import count_parameters, get_hardware_info, load_config

CONFIGS = [
    "small_cnn_scratch",
    "resnet50_finetune",
    "convnext_tiny_finetune",
    "small_vit_scratch_noclip",
    "vit_small_finetune_noclip",
]
NUM_CLASSES = 10
N_TRAIN_FULL = 45000  # 90% of the 50,000 CIFAR-10 training images (10% is validation)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model compute cost on one GPU")
    parser.add_argument("--configs", nargs="+", default=CONFIGS,
                        help="config names in configs/ (without .yaml)")
    parser.add_argument("--train-batch", type=int, default=32)
    parser.add_argument("--infer-batch", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10, help="untimed steps before timing")
    parser.add_argument("--iters", type=int, default=50, help="timed steps per repeat")
    parser.add_argument("--repeats", type=int, default=3, help="the median of the repeats is kept")
    parser.add_argument("--out", default=str(RESULTS_DIR / "compute_cost.csv"))
    return parser.parse_args()


def sync(device):
    # GPU work is asynchronous: wait for it before reading the clock
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def images_per_second(step, batch, device, args):
    for _ in range(args.warmup):
        step()
    sync(device)
    rates = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        for _ in range(args.iters):
            step()
        sync(device)
        rates.append(batch * args.iters / (time.perf_counter() - start))
    return statistics.median(rates)


def count_gflops(model, image_size, device):
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    # FlopCounterMode counts one multiply-add as 2 FLOPs (GFLOPs = 2 x GMACs)
    return counter.get_total_flops() / 1e9


def inference_throughput(model, image_size, device, amp, args):
    model.eval()
    x = torch.randn(args.infer_batch, 3, image_size, image_size, device=device)

    def step():
        with torch.inference_mode(), autocast(device, amp):
            model(x)

    return images_per_second(step, args.infer_batch, device, args)


def train_throughput(model, cfg, image_size, device, amp, args):
    model.train()
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(args.train_batch, 3, image_size, image_size, device=device)
    y = torch.randint(0, NUM_CLASSES, (args.train_batch,), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rate = images_per_second(step, args.train_batch, device, args)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
    del optimizer
    return rate, peak_gb


def benchmark(name, device, args):
    # weights do not change the speed, so skip downloading pretrained weights
    cfg = load_config(CONFIG_DIR / f"{name}.yaml", {"model.pretrained": False})
    image_size = cfg["data"]["image_size"]
    amp = cfg["train"].get("amp", True)

    torch.manual_seed(0)
    model = build_model(cfg["model"], NUM_CLASSES).to(device)
    total, trainable = count_parameters(model)
    gflops = count_gflops(model, image_size, device)
    infer_rate = inference_throughput(model, image_size, device, amp, args)
    train_rate, peak_gb = train_throughput(model, cfg, image_size, device, amp, args)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "experiment": name,
        "model": cfg["model"]["name"],
        "image_size": image_size,
        "params_m": total / 1e6,
        "trainable_m": trainable / 1e6,
        "gflops": gflops,
        "infer_img_s": infer_rate,
        "train_img_s": train_rate,
        "epoch_min_100pct": N_TRAIN_FULL / train_rate / 60,
        "peak_train_gb": peak_gb,
        "train_batch": args.train_batch,
        "infer_batch": args.infer_batch,
    }


def write_markdown(table, hardware, path):
    lines = [
        f"GPU: {hardware['gpu']} · torch {hardware['torch']} · timm {hardware['timm']} · "
        "random inputs, mixed precision",
        "",
        "| Model | Image size | Params (M) | GFLOPs | Inference (img/s) | Train (img/s) | "
        "Epoch at 100% (min) | Peak train GPU (GB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        peak = f"{r.peak_train_gb:.2f}" if pd.notna(r.peak_train_gb) else "-"
        lines.append(f"| {r.experiment} | {r.image_size} | {r.params_m:.2f} | {r.gflops:.2f} | "
                     f"{r.infer_img_s:.0f} | {r.train_img_s:.0f} | {r.epoch_min_100pct:.1f} | {peak} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("warning: no GPU found, the numbers will not be comparable")
    torch.backends.cudnn.benchmark = True
    hardware = get_hardware_info()
    print(f"GPU: {hardware['gpu']}")

    rows = []
    for name in args.configs:
        print(f"benchmarking {name} ...", flush=True)
        row = benchmark(name, device, args)
        row.update({"gpu": hardware["gpu"], "torch": hardware["torch"], "timm": hardware["timm"]})
        rows.append(row)
        print(f"  {row['params_m']:.2f}M params, {row['gflops']:.2f} GFLOPs, "
              f"infer {row['infer_img_s']:.0f} img/s, train {row['train_img_s']:.0f} img/s", flush=True)

    table = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    write_markdown(table, hardware, out_path.with_suffix(".md"))
    print(f"saved {out_path} and {out_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
