from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import yaml

from coffee_detector.stb.model import STBDetectHead
from coffee_detector.stb_guided.config import STBGuidedConfig
from coffee_detector.stb_guided.loss import (
    cross_head_class_scores,
    gt_bounded_cross_head_kl,
    positive_consistency_kl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stb_guided_config_freezes_s2_and_s3_modes() -> None:
    s2 = STBGuidedConfig(teacher_checkpoint="teacher.pt")
    assert s2.mode == "crosskd"
    assert s2.temperature == 2.0
    assert s2.distillation_weight == 0.50
    assert s2.minimum_teacher_gt_probability == 0.10

    s3 = STBGuidedConfig(
        mode="crosskd_af2",
        teacher_checkpoint="teacher.pt",
        af2_detection_weight=0.50,
        consistency_weight=0.25,
    )
    assert s3.mode == "crosskd_af2"


def test_s2_yaml_keeps_wav_l1_and_blocks_af2_mode() -> None:
    payload = yaml.safe_load(
        (REPO_ROOT / "configs/stb_guided/S2_WAVL1_STB1_CROSSKD.yaml").read_text()
    )
    assert payload["factorization"]["arm"] == "WAV_L1"
    assert payload["stb_guided"]["mode"] == "crosskd"
    assert "af2_detection_weight" not in payload["stb_guided"]


def test_gt_bounded_cross_head_kl_backpropagates_to_cross_path() -> None:
    cross = torch.tensor(
        [[0.2, -0.1, 0.0], [0.1, 0.0, -0.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[4.0, 0.0, -1.0], [0.0, 4.0, -1.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1])
    loss, stats = gt_bounded_cross_head_kl(
        cross,
        teacher,
        labels,
        temperature=2.0,
        minimum_gt_probability=0.10,
    )
    assert stats["teacher_correct_anchors"] == 2
    loss.backward()
    assert cross.grad is not None
    assert torch.isfinite(cross.grad).all()
    assert float(cross.grad.abs().sum()) > 0.0


def test_positive_consistency_detaches_clean_side() -> None:
    clean = torch.randn(1, 4, 3, requires_grad=True)
    shifted = torch.randn(1, 4, 3, requires_grad=True)
    mask = torch.tensor([[True, False, True, False]])
    loss = positive_consistency_kl(clean, shifted, mask, temperature=1.0)
    loss.backward()
    assert clean.grad is None
    assert shifted.grad is not None
    assert float(shifted.grad.abs().sum()) > 0.0


class _ScaleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight.view(1, -1, 1, 1)


class _FakeBaseHead(nn.Module):
    def __init__(self, channels: tuple[int, ...], nc: int) -> None:
        super().__init__()
        self.one2one = {
            "cls_head": nn.ModuleList([nn.Conv2d(channel, nc, 1) for channel in channels])
        }


def _fake_stb_head(channels: tuple[int, ...] = (8, 16, 32), nc: int = 3) -> STBDetectHead:
    # Construct only the interface needed by cross_head_class_scores without
    # invoking the production STBDetectHead constructor.
    head = STBDetectHead.__new__(STBDetectHead)
    nn.Module.__init__(head)
    head.nl = len(channels)
    head.nc = nc
    head.blocks = nn.ModuleList([_ScaleBlock(channel) for channel in channels])
    head.base_head = _FakeBaseHead(channels, nc)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def test_cross_head_path_updates_features_not_frozen_teacher() -> None:
    head = _fake_stb_head()
    features = [
        torch.randn(2, 8, 8, 8, requires_grad=True),
        torch.randn(2, 16, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
    ]
    scores = cross_head_class_scores(head, features, branch_name="one2one")
    assert scores.shape == (2, 3, 64 + 16 + 4)
    scores.square().mean().backward()
    assert all(feature.grad is not None for feature in features)
    assert all(float(feature.grad.abs().sum()) > 0.0 for feature in features)
    assert all(parameter.grad is None for parameter in head.parameters())
