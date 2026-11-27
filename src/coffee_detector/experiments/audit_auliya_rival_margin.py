"""Verify the native segmentation graph and isolate the new loss before training."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from coffee_detector.auliya_rival_margin import RIVAL_ARM, RIVAL_CONTRACT, RIVAL_PROTOCOL, RivalMarginSegmentLoss
from coffee_detector.experiments.run_auliya_official_frequency import (
    SCHEDULE, _ensure_runtime, trainer_for, validate_official_fold,
)


def _must(ok, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def preflight_rival_margin(development: Path, weights: Path, *, device: str = "cpu") -> dict:
    _ensure_runtime(weights)
    device = torch.device(f"cuda:{device}" if device.isdecimal() else device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for CPU preflight")
    data = validate_official_fold(development.resolve())
    ds = json.loads((data.parent / "official_fold_contract.json").read_text(encoding="utf-8"))
    source = YOLO(str(weights), task="segment").model
    models = []
    for arm in ("RGB", RIVAL_ARM):
        trainer = object.__new__(trainer_for(arm))
        trainer.data = {"nc": 21, "channels": 3, "names": dict(enumerate(ds["names"]))}
        trainer.args = SimpleNamespace(cls_remap=True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SCHEDULE["seed"])
            model = trainer.get_model(source.yaml, source, verbose=False)
        model.args = get_cfg(overrides={"task": "segment", "epochs": SCHEDULE["epochs"], "mask_ratio": 4})
        models.append(model.to(device).eval())
    native, candidate = models
    sa, sb = native.state_dict(), candidate.state_dict()
    _must(sa.keys() == sb.keys() and all(torch.equal(a, sb[k]) for k, a in sa.items()),
          "Candidate must not add or alter any native state tensor")
    x = torch.rand((2, 3, 96, 128), generator=torch.Generator().manual_seed(19)).to(device)
    with torch.inference_mode():
        ya, ra = native(x)
        yb, rb = candidate(x)
        for branch in ("one2many", "one2one"):
            _must(all(torch.equal(ra[branch][k], rb[branch][k]) for k in ("scores", "boxes", "mask_coefficient", "proto")),
                  f"Treatment changed native raw {branch} outputs")
        _must(all(torch.equal(a, b) for a, b in zip(ya, yb, strict=True)), "Treatment changes forward outputs")
    with TemporaryDirectory(prefix="rgbmargin1-preflight-") as temp:
        path = Path(temp) / "reload.pt"
        torch.save({"model": deepcopy(candidate).cpu(), "train_args": {"task": "segment"}}, path)
        loaded = YOLO(str(path), task="segment").model.to(device).eval()
        loaded.args = candidate.args  # trainer restores its configuration before init_criterion()
        _must(loaded.frequency_arm == RIVAL_ARM and isinstance(loaded.init_criterion().one2one, RivalMarginSegmentLoss),
              "Checkpoint reload lost the arm or training objective")
        with torch.inference_mode():
            _must(all(torch.equal(a, b) for a, b in zip(loaded(x)[0], yb, strict=True)), "Reload changed inference")

    masks = torch.zeros((2, 24, 32), dtype=torch.uint8, device=device)
    masks[:, 6:18, 8:24] = 1
    semantic = masks.clone()
    semantic[0].zero_()
    batch = {"img": x, "batch_idx": torch.tensor([0, 1], device=device),
             "cls": torch.tensor([[0.0], [1.0]], device=device),
             "bboxes": torch.tensor([[.5, .5, .5, .5], [.5, .5, .5, .5]], device=device),
             "masks": masks, "sem_masks": semantic}
    candidate.train()
    raw = candidate(x)
    default, margin = native.init_criterion(), candidate.init_criterion()
    pos, delta = {}, {}
    for branch in ("one2many", "one2one"):
        control, treatment = getattr(default, branch), getattr(margin, branch)
        t_a = control.get_assigned_targets_and_loss(raw[branch], batch)[0]
        t_b = treatment.get_assigned_targets_and_loss(raw[branch], batch)[0]
        _must(all(torch.equal(a, b) for a, b in zip(t_a, t_b, strict=True)), f"Native assignment changed in {branch}")
        count = int(t_a[0].sum())
        clean = control.loss(raw[branch], batch)[0]
        new = treatment.loss(raw[branch], batch)[0]
        det_other = [0, 1, 3, 4]  # box, segmentation, IoU/L1 or DFL, and semantic
        _must(count > 0 and torch.isfinite(new).all() and torch.equal(clean[det_other], new[det_other]),
              f"No foreground, nonfinite loss, or nonclassification loss changed in {branch}")
        extra = new[2] - clean[2]
        _must(extra > 0, f"Margin addition not active in {branch}")
        grad = torch.autograd.grad(extra, raw[branch]["scores"], retain_graph=True)[0].transpose(1, 2)
        _must(torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
              and torch.count_nonzero(grad[~t_a[0]]) == 0,
              f"Margin gradients did not stay on positive class scores in {branch}")
        pos[branch], delta[branch] = count, float(extra.detach())
    return {"protocol": RIVAL_PROTOCOL, "decision": "PASS", "device": str(device),
            "parameters_added": 0, "loss": RIVAL_CONTRACT, "native_state_exact": True,
            "same_forward_and_checkpoint_reload": True, "native_assignments_exact": True,
            "box_mask_semantic_and_dfl_losses_exact": True, "finite_foreground_only_gradients_both_heads": True,
            "synthetic_positive_counts": pos, "synthetic_cls_loss_increase": delta,
            "training_executed": False, "test_images_accessed": False}
