"""Static audit for the fair AF2 vs AF2+FFAB2 from-start experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from coffee_detector.afab import AFABConfig, AFABDetectionModel, load_afab_weights
from .model import AF2FFAConfig, AF2FFADetectHead, AF2FFADetectionModel, load_af2_ffa_weights


REPO_ROOT = Path(__file__).resolve().parents[3]
AF2_CONFIG = REPO_ROOT / "configs/afab/AF2_yolo26n_chaotic_amplitude.yaml"
FFA_CONFIGS = {
    "AF2FFAB2FS": REPO_ROOT / "configs/af2_ffa/AF2FFAB2FS_yolo26n_from_start.yaml",
    "AF2FFADCTFS": REPO_ROOT / "configs/af2_ffa/AF2FFADCTFS_yolo26n_from_start.yaml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.float() - right.float()).abs().max())


def _state_identical(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    lhs, rhs = left.state_dict(), right.state_dict()
    return lhs.keys() == rhs.keys() and all(torch.equal(lhs[key], rhs[key]) for key in lhs)


def run_af2_ffa_from_start_static_audit(
    d0_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 128,
) -> dict[str, Any]:
    """Prove same D0 start, exact zero adapter identity and box isolation."""

    from ultralytics import YOLO

    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    af2_payload = yaml.safe_load(AF2_CONFIG.read_text(encoding="utf-8")) or {}
    ffa_payloads = {
        code: yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for code, path in FFA_CONFIGS.items()
    }
    source = YOLO(str(checkpoint)).model.to(device).eval()
    if type(source.model[-1]).__name__ != "Detect":
        raise RuntimeError("Checkpoint D0 harus memakai native Detect")
    nc = int(source.model[-1].nc)
    afab = AFABConfig.from_mapping(af2_payload["afab"])
    af2 = AFABDetectionModel(
        str(REPO_ROOT / af2_payload["model"]), nc=nc, verbose=False, afab=afab
    ).to(device)
    load_afab_weights(af2, source)
    af2.eval()

    torch.manual_seed(20260823)
    image = torch.rand(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        af2_output = af2(image)

    records: dict[str, Any] = {}
    source_params = sum(p.numel() for p in af2.parameters())
    for code, payload in ffa_payloads.items():
        config = AF2FFAConfig.from_mapping(payload["af2_ffa"])
        candidate = AF2FFADetectionModel(
            str(REPO_ROOT / payload["model"]),
            nc=nc,
            verbose=False,
            afab=AFABConfig.from_mapping(payload["afab"]),
            af2_ffa=config,
        ).to(device)
        load_af2_ffa_weights(candidate, source)
        candidate.eval()
        head = candidate.model[-1]
        if not isinstance(head, AF2FFADetectHead):
            raise TypeError(type(head).__name__)
        with torch.inference_mode():
            zero_output = candidate(image)
        box_diff = _max_abs(
            zero_output[1]["one2one"]["boxes"], af2_output[1]["one2one"]["boxes"]
        )
        score_diff = _max_abs(
            zero_output[1]["one2one"]["scores"], af2_output[1]["one2one"]["scores"]
        )

        sizes = (16, 8, 4)
        features = [
            torch.rand(1, adapter.channels, size, size, device=device)
            for adapter, size in zip(head.adapters, sizes)
        ]
        with torch.inference_mode():
            native_head = head.base_head([item.clone() for item in features])
            wrapped_zero = head([item.clone() for item in features])
        exact_boxes = torch.equal(
            native_head[1]["one2one"]["boxes"], wrapped_zero[1]["one2one"]["boxes"]
        )
        exact_scores = torch.equal(
            native_head[1]["one2one"]["scores"], wrapped_zero[1]["one2one"]["scores"]
        )
        adapter_identity = all(
            torch.equal(adapter(feature), feature)
            for adapter, feature in zip(head.adapters, features)
        )

        descriptor = head.adapters[0].spectral_descriptor(features[0])
        descriptor_ok = (
            tuple(descriptor.shape) == (1, head.adapters[0].channels)
            and bool(torch.isfinite(descriptor).all())
            and bool((descriptor >= 0).all())
            and bool((descriptor <= 1).all())
            and bool(descriptor.abs().sum() > 0)
        )
        with torch.no_grad():
            for adapter in head.adapters:
                adapter.alpha.fill_(0.05)
                adapter.bias.fill_(0.25)
        with torch.inference_mode():
            wrapped_active = head([item.clone() for item in features])
        active_boxes_equal = torch.equal(
            wrapped_zero[1]["one2one"]["boxes"], wrapped_active[1]["one2one"]["boxes"]
        )
        active_score_change = float(
            (
                wrapped_zero[1]["one2one"]["scores"]
                - wrapped_active[1]["one2one"]["scores"]
            ).abs().max()
        )
        parameters = sum(p.numel() for p in candidate.parameters())
        added = parameters - source_params
        records[code] = {
            "descriptor_type": config.descriptor_type,
            "parameters": parameters,
            "added_parameters_vs_af2": added,
            "full_model_box_max_abs_diff": box_diff,
            "full_model_score_max_abs_diff": score_diff,
            "descriptor_min": float(descriptor.min()),
            "descriptor_max": float(descriptor.max()),
            "gates": {
                "native_head_state_preserved": _state_identical(head.base_head, source.model[-1]),
                "adapter_identity_bitwise_equal": adapter_identity,
                "same_head_boxes_bitwise_equal": exact_boxes,
                "same_head_scores_bitwise_equal": exact_scores,
                "full_model_numerically_consistent": max(box_diff, score_diff) <= 1.0e-4,
                "descriptor_finite_bounded_nonzero": descriptor_ok,
                "active_preserves_boxes_bitwise": active_boxes_equal,
                "active_changes_scores": active_score_change > 0.0,
                "added_parameters_under_one_percent": 0 < added < source_params * 0.01,
            },
        }

    rfft, dct = records["AF2FFAB2FS"], records["AF2FFADCTFS"]
    global_gates = {
        "af2_and_ffab2_same_model_yaml": af2_payload["model"] == ffa_payloads["AF2FFAB2FS"]["model"],
        "af2_and_ffab2_same_af2_config": af2_payload["afab"] == ffa_payloads["AF2FFAB2FS"]["afab"],
        "af2_and_ffab2_same_train_schedule": af2_payload["train"] == ffa_payloads["AF2FFAB2FS"]["train"],
        "rfft_and_dct_same_parameter_count": rfft["parameters"] == dct["parameters"],
        "rfft_and_dct_same_train_schedule": ffa_payloads["AF2FFAB2FS"]["train"] == ffa_payloads["AF2FFADCTFS"]["train"],
        "rfft_and_dct_only_descriptor_differs": {
            k: v for k, v in ffa_payloads["AF2FFAB2FS"]["af2_ffa"].items() if k != "descriptor_type"
        } == {
            k: v for k, v in ffa_payloads["AF2FFADCTFS"]["af2_ffa"].items() if k != "descriptor_type"
        },
    }
    all_gates = list(global_gates.values()) + [
        value
        for record in records.values()
        for value in record["gates"].values()
    ]
    result = {
        "format": "coffee_detector.af2_ffa.from_start_static_audit.v1",
        "d0_checkpoint": str(checkpoint),
        "d0_checkpoint_sha256": _sha256(checkpoint),
        "records": records,
        "global_gates": global_gates,
        "decision": "PASS" if all(all_gates) else "FAIL",
        "training_authorized": bool(all(all_gates)),
        "test_access_authorized": False,
    }
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
