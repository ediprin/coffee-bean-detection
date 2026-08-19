from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer
from coffee_detector.stb.model import STBDetectHead
from coffee_detector.wav1_factorization.config import WAV1FactorizationConfig
from coffee_detector.wav1_factorization.model import (
    WAV1FactorizationDetectionModel,
    load_factorization_weights,
)

from .config import STBGuidedConfig
from .loss import cross_head_class_scores
from .model import STBGuidedDetectionModel, load_stb_guided_weights


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    raise TypeError(f"Tidak menemukan tensor pada output {type(value).__name__}")


def _block_channels(block: torch.nn.Module) -> int:
    qkv = getattr(getattr(getattr(block, "wmsa", None), "attn", None), "qkv", None)
    if qkv is not None and hasattr(qkv, "in_features"):
        return int(qkv.in_features)
    norm = getattr(getattr(block, "wmsa", None), "norm1", None)
    shape = getattr(norm, "normalized_shape", None)
    if shape:
        return int(shape[-1])
    raise RuntimeError("Tidak dapat membaca channel STB block")


def static_stb_guided_audit(
    model_yaml: str | Path,
    d0_checkpoint: str | Path,
    teacher_checkpoint: str | Path,
    output_path: str | Path,
) -> dict:
    """Audit S2/S3 architecture without training or opening any test split."""

    from ultralytics import YOLO

    model_yaml = Path(model_yaml).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    teacher_checkpoint = Path(teacher_checkpoint).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    factorization = WAV1FactorizationConfig(arm="WAV_L1", eps=1.0e-8)
    guided = STBGuidedConfig(
        mode="crosskd",
        teacher_checkpoint=str(teacher_checkpoint),
    )

    d0 = YOLO(str(d0_checkpoint)).model
    reference = WAV1FactorizationDetectionModel(
        str(model_yaml),
        nc=21,
        ch=3,
        verbose=False,
        factorization=factorization,
    )
    student = STBGuidedDetectionModel(
        str(model_yaml),
        nc=21,
        ch=3,
        verbose=False,
        factorization=factorization,
        stb_guided=guided,
    )
    ref_transfer = load_factorization_weights(reference, d0)
    student_transfer = load_stb_guided_weights(student, d0)
    reference.eval()
    student.eval()

    teacher = YOLO(str(teacher_checkpoint)).model.cpu().eval()
    teacher_head = teacher.model[-1]
    teacher_is_stb1 = isinstance(teacher_head, STBDetectHead)
    if not teacher_is_stb1:
        raise TypeError(f"Teacher bukan STB1: {type(teacher_head).__name__}")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    reference_params = sum(p.numel() for p in reference.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    same_state_schema = reference.state_dict().keys() == student.state_dict().keys()
    serialized_keys = tuple(student.state_dict())
    no_teacher_serialized = not any("teacher" in key.lower() for key in serialized_keys)
    no_af2_serialized = not any("af2" in key.lower() for key in serialized_keys)

    torch.manual_seed(20260820)
    dummy = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        reference_output = _first_tensor(reference(dummy))
        student_output = _first_tensor(student(dummy))
    initial_inference_bitwise_equal = torch.equal(reference_output, student_output)

    channels = [_block_channels(block) for block in teacher_head.blocks]
    spatial = (16, 8, 4)
    features = [
        torch.randn(1, channel, size, size, requires_grad=True)
        for channel, size in zip(channels, spatial)
    ]
    cross = cross_head_class_scores(teacher_head, features, branch_name="one2one")
    cross.square().mean().backward()
    feature_gradient_reaches_student_side = all(
        feature.grad is not None
        and bool(torch.isfinite(feature.grad).all())
        and float(feature.grad.abs().sum()) > 0.0
        for feature in features
    )
    teacher_has_no_grad = all(parameter.grad is None for parameter in teacher.parameters())

    af2 = AFABInputEnhancer(
        AFABConfig(
            mode="af2",
            patch_size=32,
            overlap=0.50,
            gamma=0.10,
            angular_bins=360,
            chunk_size=128,
            eps=1.0e-8,
        )
    ).eval()
    af2_parameters = sum(parameter.numel() for parameter in af2.parameters())
    with torch.no_grad():
        shifted = af2(dummy)
    af2_shape_preserved = shifted.shape == dummy.shape
    af2_finite = bool(torch.isfinite(shifted).all())

    checks = {
        "teacher_is_stb1": teacher_is_stb1,
        "student_is_wav_l1": student.factorization_config.arm == "WAV_L1",
        "student_parameter_count_equals_wav_l1": student_params == reference_params,
        "student_state_schema_equals_wav_l1": bool(same_state_schema),
        "student_checkpoint_has_no_teacher": no_teacher_serialized,
        "student_checkpoint_has_no_af2": no_af2_serialized,
        "initial_inference_bitwise_equals_wav_l1_control": initial_inference_bitwise_equal,
        "cross_head_gradient_reaches_student_features": feature_gradient_reaches_student_side,
        "frozen_teacher_receives_no_parameter_grad": teacher_has_no_grad,
        "af2_training_view_has_zero_parameters": af2_parameters == 0,
        "af2_training_view_preserves_shape": af2_shape_preserved,
        "af2_training_view_is_finite": af2_finite,
    }
    result = {
        "format": "coffee_detector.stb_guided.static_audit.v1",
        "date": "2026-08-20",
        "scope": "architecture-only; no training; no test access",
        "factorization": factorization.to_dict(),
        "guided": {**guided.to_dict(), "teacher_checkpoint": str(teacher_checkpoint)},
        "d0_checkpoint_sha256": sha256(d0_checkpoint),
        "teacher_checkpoint_sha256": sha256(teacher_checkpoint),
        "parameters": {
            "wav_l1_reference": reference_params,
            "stb_guided_student": student_params,
            "af2_training_view": af2_parameters,
        },
        "transfer": {
            "wav_l1_reference": ref_transfer,
            "stb_guided_student": student_transfer,
        },
        "checks": checks,
        "test_images_accessed": False,
        "test_opened": False,
        "decision": "PASS" if all(checks.values()) else "FAIL",
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
