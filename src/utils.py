import copy
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def deep_merge(base, child):
    merged = copy.deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_overrides(pairs):
    overrides = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override must look like key=value, got: {pair}")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = parse_value(value)
    return overrides


def parse_value(text):
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def set_by_dotted_key(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def load_config(path, overrides=None):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent_name = cfg.pop("inherit", None)
    if parent_name:
        parent = load_config(path.parent / parent_name)
        cfg = deep_merge(parent, cfg)

    for key, value in (overrides or {}).items():
        set_by_dotted_key(cfg, key, value)
    return cfg


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_hardware_info():
    import timm

    info = {
        "cpu": platform.processor() or platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "cuda": torch.version.cuda,
        "gpu": None,
        "gpu_memory_gb": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
    return info


def run_name(cfg):
    data = cfg["data"]
    return f"frac{data['train_fraction']}_aug{data['augmentation']}_seed{cfg['seed']}"


def make_run_dir(cfg):
    run_dir = Path(cfg["output_dir"]) / cfg["experiment"] / run_name(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return run_dir
