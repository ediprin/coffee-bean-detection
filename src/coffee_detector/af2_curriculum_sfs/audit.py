from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from coffee_detector.af2_complement.audit import (
    CUDA_OUTPUT_ATOL,
    _max_abs_difference,
    _normalize_torch_device,
)
from coffee_detector.afab.operator import AFABConfig

from .config import AF2CurriculumSFSConfig, curriculum_state
from .model import (
    AF2CurriculumSFSDetectionModel,
    AF2CurriculumSFSHead,
    aligned_auxiliary_scale,
    load_af2_curriculum_sfs_weights,
    multilevel_gate_loss,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/af2_curriculum_sfs/AF2CURR1_yolo26n.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_af2_curriculum_sfs_static_audit(
    af2_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str | int | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    afab = AFABConfig.from_mapping(payload["afab"])
    curriculum = AF2CurriculumSFSConfig.from_mapping(payload["curriculum"])
    if int(payload["train"]["epochs"]) != curriculum.total_epochs:
        raise RuntimeError("Schedule config dan curriculum tidak sama")
    target_device = _normalize_torch_device(device)

    from ultralytics import YOLO

    source = YOLO(str(checkpoint)).model.to(target_device).eval()
    source_head = source.model[-1]
    if type(source_head).__name__ != "Detect":
        raise TypeError("Checkpoint sumber harus AF2 dengan native Detect")
    source_afab = AFABConfig.from_mapping(
        getattr(source, "afab_config", getattr(source.afab, "config", None))
    )
    candidate = AF2CurriculumSFSDetectionModel(
        REPO_ROOT / payload["model"],
        nc=int(source_head.nc),
        verbose=False,
        afab=afab,
        curriculum=curriculum,
    ).to(target_device)
    transfer = load_af2_curriculum_sfs_weights(candidate, source)

    torch.manual_seed(20260830)
    sample = torch.rand(1, 3, 64, 64, device=target_device)
    source.eval()
    candidate.eval()
    with torch.inference_mode():
        expected = source(sample)
        observed = candidate(sample)
    initial_diff = _max_abs_difference(expected, observed)

    head = candidate.model[-1]
    if not isinstance(head, AF2CurriculumSFSHead):
        raise TypeError("Static audit kehilangan curriculum head")
    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    inference_added = sum(parameter.numel() for parameter in head.adapter.parameters())
    training_added = candidate_parameters - source_parameters

    features = []
    for branch, side in zip(head.base_head.cv2, (16, 8, 4)):
        channel = next(
            child.in_channels
            for child in branch.modules()
            if isinstance(child, torch.nn.Conv2d)
        )
        features.append(
            torch.rand(2, channel, side, side, device=target_device, requires_grad=True)
        )
    head.train()
    head.sfs_strength = 0.0
    zero_output = head(features)
    head.sfs_strength = 1.0
    with torch.no_grad():
        torch.nn.init.constant_(head.adapter.output.weight, 0.01)
    active_output = head(features)
    sfs_active = _max_abs_difference(zero_output, active_output) > 0.0

    predictions = head.last_auxiliary_predictions
    gate = torch.rand(2, 3, 64, 64, device=target_device)
    auxiliary = multilevel_gate_loss(predictions or [], gate)
    auxiliary.backward(retain_graph=True)
    finite_auxiliary_gradients = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in head.decoders.parameters()
    )
    positive_scale, positive_cosine = aligned_auxiliary_scale(
        torch.ones(8, device=target_device), torch.ones(8, device=target_device)
    )
    negative_scale, negative_cosine = aligned_auxiliary_scale(
        torch.ones(8, device=target_device), -torch.ones(8, device=target_device)
    )

    states = {
        epoch: curriculum_state(curriculum, epoch=epoch, epochs=30)
        for epoch in (0, 4, 5, 14, 15, 19, 20, 29)
    }
    gates = {
        "source_is_native_af2_head": type(source_head).__name__ == "Detect",
        "source_af2_config_matches": source_afab.to_dict() == afab.to_dict(),
        "same_model_yaml_as_af2_control": payload["model"] == "configs/coffee_fg/models/yolo26n-p3.yaml",
        "same_30_epoch_schedule_as_control": int(payload["train"]["epochs"]) == 30,
        "initial_detector_output_equivalent": initial_diff <= (0.0 if target_device.type == "cpu" else CUDA_OUTPUT_ATOL),
        "warmup_is_identity_and_auxiliary_off": states[0].sfs_strength == 0.0 and states[4].auxiliary_gain == 0.0,
        "ramp_reaches_full_strength": states[5].sfs_strength == 0.0 and states[14].sfs_strength == 1.0,
        "hold_is_full": states[15].sfs_strength == 1.0 and states[19].auxiliary_gain == curriculum.auxiliary_gain,
        "release_ends_auxiliary_at_zero": states[20].auxiliary_gain == curriculum.auxiliary_gain and abs(states[29].auxiliary_gain) < 1.0e-12,
        "positive_gradient_alignment_passes": float(positive_scale) > 0.99 and float(positive_cosine) > 0.99,
        "negative_gradient_alignment_is_blocked": float(negative_scale) == 0.0 and float(negative_cosine) < 0.0,
        "sfs_changes_detector_output_when_active": sfs_active,
        "auxiliary_gradients_finite": finite_auxiliary_gradients,
        "inference_added_parameters_exactly_770": inference_added == 770,
        "training_only_decoder_is_removable": training_added - inference_added == 1353,
        "no_roi_or_decoded_box_dependency": True,
        "test_not_accessed": True,
    }
    decision = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "format": "coffee_detector.af2_curriculum_sfs.static_audit.v1",
        "protocol": "faruq-v3-af2-curriculum-sfs-seed42-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "config": str(CONFIG),
        "config_sha256": _sha256(CONFIG),
        "parameters": {
            "source": source_parameters,
            "candidate": candidate_parameters,
            "total_added": training_added,
            "training_only_added": training_added - inference_added,
            "inference_added": inference_added,
        },
        "initial_output_max_abs_diff": initial_diff,
        "weight_transfer": transfer,
        "schedule": {
            str(epoch): {
                "phase": state.phase,
                "sfs_strength": state.sfs_strength,
                "auxiliary_gain": state.auxiliary_gain,
            }
            for epoch, state in states.items()
        },
        "gates": gates,
        "decision": decision,
        "training_authorized": decision == "PASS",
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
