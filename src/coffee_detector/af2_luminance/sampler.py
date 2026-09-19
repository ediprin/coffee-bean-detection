from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import torch
from torch.utils.data import Sampler


_IDENTITY_STEM = re.compile(r"^([0-9a-f]{64})(?:_[0-9]{2})?$")


class EpochWeightedSampler(Sampler[int]):
    """Deterministic weighted sampler whose sequence is keyed by epoch."""

    def __init__(self, weights: torch.Tensor, num_samples: int, *, seed: int) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double).cpu()
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        if self.weights.ndim != 1 or len(self.weights) == 0:
            raise ValueError("Sampler weights harus vektor non-empty")
        if self.num_samples <= 0:
            raise ValueError("num_samples harus positif")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indexes = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=generator
        )
        return iter(indexes.tolist())

    def __len__(self) -> int:
        return self.num_samples


def j25_source_identity(path: str | Path) -> str:
    """Recover the frozen J25 source identity from an extracted image name."""

    stem = Path(path).stem
    match = _IDENTITY_STEM.fullmatch(stem)
    if match is None:
        raise ValueError(f"Nama image J25 tidak memuat source identity: {path}")
    return match.group(1)


def identity_repeat_factor_weights(
    image_paths: Sequence[str | Path],
    class_sets: Sequence[Iterable[int]],
    *,
    class_count: int,
    maximum_repeat: float = 4.0,
) -> tuple[torch.Tensor, dict]:
    """Build LVIS-style repeat weights at source-identity level.

    Derivative siblings split one identity's weight equally.  The repeat
    threshold is the median positive train-only class frequency, so no
    validation statistic or hand-selected class list enters the sampler.
    """

    if len(image_paths) != len(class_sets) or not image_paths:
        raise ValueError("Image path dan class set harus non-empty dan sejajar")
    if class_count <= 0 or maximum_repeat < 1:
        raise ValueError("Konfigurasi repeat-factor tidak valid")

    members: dict[str, list[int]] = defaultdict(list)
    identity_classes: dict[str, set[int]] = defaultdict(set)
    for index, (path, classes) in enumerate(zip(image_paths, class_sets)):
        identity = j25_source_identity(path)
        members[identity].append(index)
        for value in classes:
            class_id = int(value)
            if not 0 <= class_id < class_count:
                raise ValueError(f"Class ID di luar ontologi: {class_id}")
            identity_classes[identity].add(class_id)

    identity_count = len(members)
    class_identity_counts = Counter(
        class_id for values in identity_classes.values() for class_id in values
    )
    if set(class_identity_counts) != set(range(class_count)):
        missing = sorted(set(range(class_count)) - set(class_identity_counts))
        raise RuntimeError(f"Train identity kehilangan kelas: {missing}")
    frequencies = {
        class_id: class_identity_counts[class_id] / identity_count
        for class_id in range(class_count)
    }
    threshold = float(median(frequencies.values()))
    class_repeats = {
        class_id: min(maximum_repeat, max(1.0, math.sqrt(threshold / frequency)))
        for class_id, frequency in frequencies.items()
    }
    identity_repeats = {
        identity: max(class_repeats[class_id] for class_id in classes)
        for identity, classes in identity_classes.items()
    }
    weights = torch.empty(len(image_paths), dtype=torch.double)
    for identity, indexes in members.items():
        derivative_weight = identity_repeats[identity] / len(indexes)
        for index in indexes:
            weights[index] = derivative_weight
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise RuntimeError("Repeat-factor weights tidak finite/positif")

    aggregate = {
        identity: float(weights[indexes].sum()) for identity, indexes in members.items()
    }
    if any(
        not math.isclose(aggregate[identity], repeat, rel_tol=1e-12, abs_tol=1e-12)
        for identity, repeat in identity_repeats.items()
    ):
        raise RuntimeError("Bobot derivative tidak kembali ke bobot identity")
    summary = {
        "format": "coffee_detector.j25.identity_repeat_factor.v1",
        "images": len(image_paths),
        "source_identities": identity_count,
        "class_count": class_count,
        "threshold_source": "median_positive_train_identity_frequency",
        "threshold": threshold,
        "maximum_repeat": maximum_repeat,
        "class_identity_counts": {
            str(key): int(class_identity_counts[key]) for key in range(class_count)
        },
        "class_repeat_factors": {
            str(key): float(class_repeats[key]) for key in range(class_count)
        },
        "identity_repeat_min": float(min(identity_repeats.values())),
        "identity_repeat_max": float(max(identity_repeats.values())),
        "derivative_weight_min": float(weights.min()),
        "derivative_weight_max": float(weights.max()),
    }
    return weights, summary
