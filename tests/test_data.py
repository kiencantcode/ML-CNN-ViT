import numpy as np
import pytest

from src.data import (
    build_transforms,
    cifar_data_config,
    make_train_val_indices,
    subsample_train_indices,
)

FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]


@pytest.fixture(scope="module")
def labels():
    rng = np.random.default_rng(123)
    return np.repeat(np.arange(10), 5000)[rng.permutation(50000)]


@pytest.fixture(scope="module")
def split(labels):
    return make_train_val_indices(labels, val_fraction=0.1, seed=0)


def test_train_val_do_not_overlap(split):
    train_idx, val_idx = split
    assert len(train_idx) == 45000
    assert len(val_idx) == 5000
    assert len(set(train_idx) & set(val_idx)) == 0


def test_split_is_repeatable(labels, split):
    train_again, val_again = make_train_val_indices(labels, val_fraction=0.1, seed=0)
    assert np.array_equal(split[0], train_again)
    assert np.array_equal(split[1], val_again)


def test_validation_is_stratified(labels, split):
    _, val_idx = split
    assert np.all(np.bincount(labels[val_idx], minlength=10) == 500)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_subsets_are_nested(labels, split, seed):
    train_idx, _ = split
    subsets = [set(subsample_train_indices(train_idx, labels, f, seed)) for f in FRACTIONS]
    for smaller, larger in zip(subsets, subsets[1:]):
        assert smaller <= larger


@pytest.mark.parametrize("fraction", FRACTIONS)
def test_subsets_are_stratified(labels, split, fraction):
    train_idx, _ = split
    subset = subsample_train_indices(train_idx, labels, fraction, seed=0)
    counts = np.bincount(labels[subset], minlength=10)
    expected = fraction * 4500
    assert np.all(np.abs(counts - expected) <= 1)


def test_subsets_stay_inside_train(labels, split):
    train_idx, val_idx = split
    subset = subsample_train_indices(train_idx, labels, 0.1, seed=0)
    assert set(subset) <= set(train_idx)
    assert len(set(subset) & set(val_idx)) == 0


def test_seed_changes_subset(labels, split):
    train_idx, _ = split
    a = subsample_train_indices(train_idx, labels, 0.1, seed=0)
    b = subsample_train_indices(train_idx, labels, 0.1, seed=1)
    assert not np.array_equal(a, b)


def test_eval_transform_is_deterministic():
    from PIL import Image

    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8))
    tf = build_transforms({}, cifar_data_config(32), "strong", is_train=False)
    assert tf(image).equal(tf(image))


def test_transform_output_size():
    from PIL import Image

    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    model_cfg = {"input_size": (3, 224, 224), "mean": (0.5,) * 3, "std": (0.5,) * 3,
                 "interpolation": "bicubic"}
    for aug in ["none", "basic", "strong"]:
        tf = build_transforms({}, model_cfg, aug, is_train=True)
        assert tuple(tf(image).shape) == (3, 224, 224)
