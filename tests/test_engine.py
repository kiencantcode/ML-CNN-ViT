"""Smoke tests for the optional training features in src/engine.py:
last.pt, resume, early stopping, gradient clipping. Tiny model on random data, CPU only.

Run: pytest tests/test_engine.py
"""

import csv
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import src.engine as engine


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 16), nn.ReLU())
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        return self.head(self.body(x))

    def get_classifier(self):
        return self.head


def _cfg(epochs=3, **train_overrides):
    train = {
        "epochs": epochs, "batch_size": 16, "optimizer": "adamw",
        "lr_head": 1e-2, "lr_backbone": 1e-3, "weight_decay": 0.0,
        "warmup_epochs": 1, "label_smoothing": 0.0, "amp": False, "grad_accum_steps": 1,
    }
    train.update(train_overrides)
    return {
        "seed": 0, "experiment": "smoke", "progress_bar": False,
        "model": {"name": "tiny", "pretrained": False},
        "data": {"train_fraction": 1.0, "augmentation": "none"},
        "train": train,
    }


def _loaders():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(96, 3, 8, 8, generator=gen)
    y = torch.randint(0, 10, (96,), generator=gen)
    ds = TensorDataset(x, y)
    return (DataLoader(ds, batch_size=16, shuffle=True),
            DataLoader(ds, batch_size=32), DataLoader(ds, batch_size=32))


def _fit(tmp_path, cfg, resume=False, model=None):
    torch.manual_seed(0)
    model = model or TinyNet()
    return engine.fit(model, _loaders(), cfg, tmp_path, torch.device("cpu"), resume=resume)


def _history(path):
    with open(path / "history.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_default_config_writes_no_last_pt(tmp_path):
    # Configs that do not set the new keys (the CNN configs) must behave as before.
    metrics = _fit(tmp_path, _cfg(epochs=2))
    assert (tmp_path / "best.pt").exists()
    assert not (tmp_path / "last.pt").exists()
    assert metrics["epochs_run"] == 2 and metrics["early_stopped"] is False
    assert len(_history(tmp_path)) == 2


def test_last_pt_holds_full_training_state(tmp_path):
    _fit(tmp_path, _cfg(epochs=2, save_last=True))
    ckpt = torch.load(tmp_path / "last.pt", weights_only=False)
    assert ckpt["epoch"] == 2
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "epoch_times"):
        assert key in ckpt


def test_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    full_dir, cut_dir = tmp_path / "full", tmp_path / "cut"
    full_dir.mkdir()
    cut_dir.mkdir()
    cfg = _cfg(epochs=4, save_last=True)
    full = _fit(full_dir, cfg)

    # Simulate a Colab disconnect: the third epoch crashes before it starts.
    real_train_one_epoch = engine.train_one_epoch
    calls = {"n": 0}

    def crash_on_third_epoch(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated disconnect")
        return real_train_one_epoch(*args, **kwargs)

    monkeypatch.setattr(engine, "train_one_epoch", crash_on_third_epoch)
    try:
        _fit(cut_dir, cfg)
    except RuntimeError:
        pass
    monkeypatch.setattr(engine, "train_one_epoch", real_train_one_epoch)
    assert torch.load(cut_dir / "last.pt", weights_only=False)["epoch"] == 2

    resumed = _fit(cut_dir, cfg, resume=True)
    assert resumed["epochs_run"] == 4
    assert len(_history(cut_dir)) == 4
    assert abs(resumed["best_val_acc"] - full["best_val_acc"]) < 1e-6
    assert abs(resumed["test_acc"] - full["test_acc"]) < 1e-6


def test_resume_without_last_pt_fails_clearly(tmp_path):
    try:
        _fit(tmp_path, _cfg(epochs=2), resume=True)
    except FileNotFoundError as err:
        assert "last.pt" in str(err)
    else:
        raise AssertionError("resume without last.pt should raise")


def test_early_stopping_stops_and_is_recorded(tmp_path):
    # lr 0 means val accuracy never improves after epoch 1, so patience 2 stops at epoch 3.
    cfg = _cfg(epochs=10, lr_head=0.0, lr_backbone=0.0, early_stopping_patience=2)
    metrics = _fit(tmp_path, cfg)
    assert metrics["early_stopped"] is True
    assert metrics["epochs_run"] == 3
    assert metrics["best_epoch"] == 1
    assert len(_history(tmp_path)) == 3
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["early_stopping_patience"] == 2


def test_grad_clip_runs(tmp_path):
    metrics = _fit(tmp_path, _cfg(epochs=1, grad_clip=1.0))
    assert metrics["epochs_run"] == 1
