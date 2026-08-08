from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn

from coffee_detector.dsr_sscb.loss import rasterize_bbox_foreground, semantic_foreground_loss
from coffee_detector.dsr_sscb.model import (
    SSCBClassificationPath,
    SSCBConfig,
    SSCBDetectHead,
)


ROOT = Path(__file__).resolve().parents[1]


def test_config_modes_and_flags():
    m0 = SSCBConfig.from_mapping({"mode": "msda"})
    s0 = SSCBConfig.from_mapping({"mode": "semantic_aux"})
    s1 = SSCBConfig.from_mapping({"mode": "calibrated"})
    assert not m0.uses_semantics
    assert s0.uses_semantics and not s0.uses_calibration
    assert s1.uses_semantics and s1.uses_calibration


def test_repaired_screening_uses_same_memory_safe_batch_for_all_arms():
    paths = [
        ROOT / "configs/dsr_sscb/M0_msda.yaml",
        ROOT / "configs/dsr_sscb/S0_semantic_aux_msda.yaml",
        ROOT / "configs/dsr_sscb/S1_calibrated_sscb.yaml",
    ]
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["train"]["batch"] == 8
        assert payload["train"]["epochs"] == 50
        assert payload["train"]["imgsz"] == 640


def test_bbox_rasterization_and_semantic_loss_are_finite():
    batch_idx = torch.tensor([0.0, 0.0, 1.0])
    boxes = torch.tensor(
        [
            [0.5, 0.5, 0.4, 0.4],
            [0.2, 0.2, 0.1, 0.1],
            [0.7, 0.7, 0.2, 0.2],
        ]
    )
    target = rasterize_bbox_foreground(
        batch_idx,
        boxes,
        batch_size=2,
        height=20,
        width=20,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert target.shape == (2, 1, 20, 20)
    assert target.sum() > 0
    logits = [torch.zeros(2, 1, 20, 20), torch.zeros(2, 1, 10, 10)]
    loss = semantic_foreground_loss(logits, {"batch_idx": batch_idx, "bboxes": boxes})
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_sscb_classification_path_shapes_and_zero_start():
    config = SSCBConfig(mode="semantic_aux", hidden_dim=8, sampling_points=1, max_offset_pixels=1.0)
    path = SSCBClassificationPath((8, 16, 32), num_classes=5, config=config)
    features = [
        torch.randn(2, 8, 8, 8),
        torch.randn(2, 16, 4, 4),
        torch.randn(2, 32, 2, 2),
    ]
    corrections, semantic_logits, diagnostics = path(features)
    assert [tuple(x.shape) for x in corrections] == [(2, 5, 8, 8), (2, 5, 4, 4), (2, 5, 2, 2)]
    assert [tuple(x.shape) for x in semantic_logits] == [(2, 1, 8, 8), (2, 1, 4, 4), (2, 1, 2, 2)]
    assert all(torch.count_nonzero(x) == 0 for x in corrections)
    assert set(diagnostics) == {"level0", "level1", "level2"}


class _ConvBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class _FakeDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.nc = 3
        self.nl = 3
        self.reg_max = 1
        self.end2end = True
        self.max_det = 300
        self.export = False
        self.format = None
        self.dynamic = False
        self.agnostic_nms = False
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        channels = (4, 8, 16)
        self.cv2 = nn.ModuleList([_ConvBranch(c, 4) for c in channels])
        self.one2many = nn.ModuleDict(
            {
                "box_head": nn.ModuleList([_ConvBranch(c, 4) for c in channels]),
                "cls_head": nn.ModuleList([_ConvBranch(c, self.nc) for c in channels]),
            }
        )
        self.one2one = nn.ModuleDict(
            {
                "box_head": nn.ModuleList([_ConvBranch(c, 4) for c in channels]),
                "cls_head": nn.ModuleList([_ConvBranch(c, self.nc) for c in channels]),
            }
        )


_FakeDetect.__name__ = "Detect"


def test_training_invokes_sscb_only_once_for_one2many(monkeypatch):
    base = _FakeDetect()
    head = SSCBDetectHead(base, SSCBConfig(mode="semantic_aux", hidden_dim=4, sampling_points=1))
    features = [
        torch.randn(2, 4, 8, 8),
        torch.randn(2, 8, 4, 4),
        torch.randn(2, 16, 2, 2),
    ]
    calls = {"count": 0}
    original = head.sscb.forward

    def counted(features):
        calls["count"] += 1
        return original(features)

    monkeypatch.setattr(head.sscb, "forward", counted)
    head.train()
    out = head(features)
    assert calls["count"] == 1
    assert "sscb_semantic_logits" in out["one2many"]
    assert "sscb_semantic_logits" not in out["one2one"]


def test_one2one_scores_are_native_when_one2many_sscb_correction_is_nonzero():
    base = _FakeDetect()
    head = SSCBDetectHead(base, SSCBConfig(mode="semantic_aux", hidden_dim=4, sampling_points=1))
    for correction in head.sscb.class_corrections:
        nn.init.constant_(correction.bias, 1.0)
    features = [
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 8, 4, 4),
        torch.randn(1, 16, 2, 2),
    ]
    head.train()
    out = head(features)
    native_one2one = head._native_branch([x.detach() for x in features], head.one2one)
    assert torch.allclose(out["one2one"]["scores"], native_one2one["scores"])
