"""Static causal/safety audit for AF2 feature-frequency adapters."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from coffee_detector.afab import AFABConfig
from .model import (
    AF2FFAConfig,
    AF2FFADetectHead,
    AF2FFADetectionModel,
    load_af2_ffa_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "AF2FFA0": REPO_ROOT / "configs/af2_ffa/AF2FFA0_yolo26n_zero_control.yaml",
    "AF2FFA1": REPO_ROOT / "configs/af2_ffa/AF2FFA1_yolo26n_spectral_adapter.yaml",
    "AF2FFAB1": REPO_ROOT / "configs/af2_ffa/AF2FFAB1_yolo26n_bounded_spectral_adapter.yaml",
    "AF2FFAB2": REPO_ROOT / "configs/af2_ffa/AF2FFAB2_yolo26n_gradient_matched_bounded_adapter.yaml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_nonzero(parameters) -> bool:
    gradients = [p.grad for p in parameters if p.grad is not None]
    return bool(gradients) and all(torch.isfinite(g).all() for g in gradients) and any(
        bool(g.abs().max() > 0) for g in gradients
    )


def _max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.float() - right.float()).abs().max())


def _state_identical(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    lhs, rhs = left.state_dict(), right.state_dict()
    return lhs.keys() == rhs.keys() and all(torch.equal(lhs[key], rhs[key]) for key in lhs)


def run_af2_ffa_static_audit(
    af2_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 128,
) -> dict[str, Any]:
    from ultralytics import YOLO

    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    configs = {
        code: yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for code, path in CONFIGS.items()
    }
    source = YOLO(str(checkpoint)).model.to(device).eval()
    if getattr(source, "afab", None) is None:
        raise RuntimeError("Checkpoint sumber bukan AF2")
    nc = int(source.model[-1].nc)
    torch.manual_seed(20260822)
    image = torch.rand(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        native = source(image)

    records: dict[str, Any] = {}
    models: dict[str, AF2FFADetectionModel] = {}
    for code, payload in configs.items():
        candidate = AF2FFADetectionModel(
            str(REPO_ROOT / payload["model"]),
            nc=nc,
            verbose=False,
            afab=AFABConfig.from_mapping(payload["afab"]),
            af2_ffa=AF2FFAConfig.from_mapping(payload["af2_ffa"]),
        ).to(device)
        transfer = load_af2_ffa_weights(candidate, source)
        candidate.eval()
        head = candidate.model[-1]
        if not isinstance(head, AF2FFADetectHead):
            raise TypeError(type(head).__name__)
        with torch.inference_mode():
            zero = candidate(image)
        full_box_diff = _max_abs_difference(
            zero[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"]
        )
        full_score_diff = _max_abs_difference(
            zero[1]["one2one"]["scores"], native[1]["one2one"]["scores"]
        )

        # Prove exact identity on the same head and the same feature tensors.
        # Separate end-to-end CUDA forwards include FFT kernels and can differ
        # by a few ULPs even when the wiring and weights are identical.
        identity_features = [
            torch.rand(1, adapter.channels, size, size, device=device)
            for adapter, size in zip(head.adapters, (16, 8, 4))
        ]
        with torch.inference_mode():
            native_head = head.base_head(
                [feature.clone() for feature in identity_features]
            )
            wrapped_head = head([feature.clone() for feature in identity_features])
        zero_boxes = torch.equal(
            wrapped_head[1]["one2one"]["boxes"],
            native_head[1]["one2one"]["boxes"],
        )
        zero_scores = torch.equal(
            wrapped_head[1]["one2one"]["scores"],
            native_head[1]["one2one"]["scores"],
        )
        adapter_identity = all(
            torch.equal(adapter(feature), feature)
            for adapter, feature in zip(head.adapters, identity_features)
        )

        active = AF2FFADetectionModel(
            str(REPO_ROOT / payload["model"]),
            nc=nc,
            verbose=False,
            afab=AFABConfig.from_mapping(payload["afab"]),
            af2_ffa=AF2FFAConfig.from_mapping(payload["af2_ffa"]),
        ).to(device)
        load_af2_ffa_weights(active, source)
        # A freshly constructed synthetic checkpoint can have a class tower
        # whose output is temporarily bias-only.  Probe the wiring with a
        # deterministic non-degenerate class tower while retaining the loaded
        # box tower exactly.
        with torch.no_grad():
            torch.manual_seed(73)
            for parameter in active.model[-1].base_head.one2one["cls_head"].parameters():
                parameter.normal_(0.0, 0.05)
        active.eval()
        active_head = active.model[-1]
        synthetic_features = [
            torch.rand(1, adapter.channels, size, size, device=device)
            for adapter, size in zip(active_head.adapters, (16, 8, 4))
        ]
        with torch.inference_mode():
            active_reference = active_head(
                [feature.clone() for feature in synthetic_features]
            )
        with torch.no_grad():
            for adapter in active.model[-1].adapters:
                adapter.alpha.fill_(0.25)
                adapter.bias.fill_(0.25)
        with torch.inference_mode():
            changed = active_head([feature.clone() for feature in synthetic_features])
        active_boxes = torch.equal(
            changed[1]["one2one"]["boxes"],
            active_reference[1]["one2one"]["boxes"],
        )
        active_score_diff = float(
            (
                changed[1]["one2one"]["scores"]
                - active_reference[1]["one2one"]["scores"]
            )
            .abs()
            .max()
        )

        probe = active.train()
        probe.zero_grad(set_to_none=True)
        scores = probe(image)["one2many"]["scores"]
        scores.square().mean().backward()
        head_probe = probe.model[-1]
        descriptor_sum = 0.0
        with torch.no_grad():
            sample_feature = torch.rand(
                1, head_probe.adapters[0].channels, 16, 16, device=device
            )
            descriptor_sum = float(
                head_probe.adapters[0].spectral_descriptor(sample_feature).abs().sum()
            )
        cap = head_probe.adapters[0].config.residual_gain_cap
        amplitude_gradient = torch.autograd.grad(
            head.adapters[0].residual_amplitude().sum(),
            head.adapters[0].alpha,
            retain_graph=False,
        )[0]
        amplitude_gradient_mean = float(amplitude_gradient.mean())
        bounded_deviation = 0.0
        if cap is not None:
            bounded_adapter = head_probe.adapters[0]
            bounded_input = torch.ones(
                1, bounded_adapter.channels, 16, 16, device=device
            )
            with torch.no_grad():
                bounded_adapter.alpha.fill_(100.0)
                bounded_adapter.bias.fill_(100.0)
                bounded_output = bounded_adapter(bounded_input)
            bounded_deviation = float((bounded_output - bounded_input).abs().max())
        source_code = inspect.getsource(AF2FFADetectHead)
        records[code] = {
            "conditioning": payload["af2_ffa"]["conditioning"],
            "residual_gain_cap": payload["af2_ffa"].get("residual_gain_cap"),
            "gradient_matched_cap": payload["af2_ffa"].get(
                "gradient_matched_cap", False
            ),
            "initial_amplitude_gradient_mean": amplitude_gradient_mean,
            "parameters": sum(p.numel() for p in candidate.parameters()),
            "state_schema": {
                key: tuple(value.shape) for key, value in candidate.state_dict().items()
            },
            "descriptor_abs_sum": descriptor_sum,
            "full_model_box_max_abs_diff": full_box_diff,
            "full_model_score_max_abs_diff": full_score_diff,
            "transfer": transfer,
            "gates": {
                "native_head_state_bitwise_preserved": _state_identical(
                    head.base_head, source.model[-1]
                ),
                "adapter_identity_bitwise_equal": adapter_identity,
                "identity_boxes_bitwise_equal": zero_boxes,
                "identity_scores_bitwise_equal": zero_scores,
                "full_model_numerically_consistent": max(
                    full_box_diff, full_score_diff
                ) <= 1.0e-4,
                "active_preserves_boxes": active_boxes,
                "active_changes_scores": active_score_diff > 0.0,
                "finite_nonzero_adapter_gradients": _finite_nonzero(
                    head_probe.adapters.parameters()
                ),
                "no_roi_align": "roi_align" not in source_code,
                "no_decoded_box_dependency": "decode" not in source_code.lower(),
                "bounded_gain_enforced": cap is None
                or bounded_deviation <= float(cap) + 1.0e-6,
            },
        }
        models[code] = candidate

    control, candidate, bounded_invalid, bounded_matched = (
        records["AF2FFA0"],
        records["AF2FFA1"],
        records["AF2FFAB1"],
        records["AF2FFAB2"],
    )
    source_parameters = sum(p.numel() for p in source.parameters())
    added = candidate["parameters"] - source_parameters
    global_gates = {
        "same_model_yaml": configs["AF2FFA0"]["model"] == configs["AF2FFA1"]["model"],
        "same_af2_config": configs["AF2FFA0"]["afab"] == configs["AF2FFA1"]["afab"],
        "same_training_schedule": configs["AF2FFA0"]["train"] == configs["AF2FFA1"]["train"],
        "only_conditioning_differs": {
            k: v for k, v in configs["AF2FFA0"]["af2_ffa"].items() if k != "conditioning"
        } == {
            k: v for k, v in configs["AF2FFA1"]["af2_ffa"].items() if k != "conditioning"
        },
        "same_parameter_count": control["parameters"] == candidate["parameters"],
        "same_state_schema": control["state_schema"] == candidate["state_schema"],
        "bounded_same_model_yaml": configs["AF2FFAB1"]["model"]
        == configs["AF2FFA1"]["model"],
        "bounded_same_af2_config": configs["AF2FFAB1"]["afab"]
        == configs["AF2FFA1"]["afab"],
        "bounded_same_training_schedule": configs["AF2FFAB1"]["train"]
        == configs["AF2FFA1"]["train"],
        "bounded_same_parameter_count": bounded_invalid["parameters"]
        == candidate["parameters"],
        "bounded_same_state_schema": bounded_invalid["state_schema"]
        == candidate["state_schema"],
        "bounded_only_adds_gain_cap": {
            key: value
            for key, value in configs["AF2FFAB1"]["af2_ffa"].items()
            if key != "residual_gain_cap"
        }
        == configs["AF2FFA1"]["af2_ffa"],
        "bounded_gain_cap_is_10_percent": configs["AF2FFAB1"]["af2_ffa"].get(
            "residual_gain_cap"
        )
        == 0.10,
        "matched_bound_same_model_yaml": configs["AF2FFAB2"]["model"]
        == configs["AF2FFA1"]["model"],
        "matched_bound_same_af2_config": configs["AF2FFAB2"]["afab"]
        == configs["AF2FFA1"]["afab"],
        "matched_bound_same_training_schedule": configs["AF2FFAB2"]["train"]
        == configs["AF2FFA1"]["train"],
        "matched_bound_same_parameter_count": bounded_matched["parameters"]
        == candidate["parameters"],
        "matched_bound_same_state_schema": bounded_matched["state_schema"]
        == candidate["state_schema"],
        "matched_bound_only_adds_cap_parameterization": {
            key: value
            for key, value in configs["AF2FFAB2"]["af2_ffa"].items()
            if key not in {"residual_gain_cap", "gradient_matched_cap"}
        }
        == configs["AF2FFA1"]["af2_ffa"],
        "matched_bound_gain_cap_is_10_percent": configs["AF2FFAB2"][
            "af2_ffa"
        ].get("residual_gain_cap")
        == 0.10,
        "matched_bound_flag_enabled": configs["AF2FFAB2"]["af2_ffa"].get(
            "gradient_matched_cap"
        )
        is True,
        "legacy_bound_gradient_confound_detected": abs(
            bounded_invalid["initial_amplitude_gradient_mean"] - 0.10
        )
        <= 1.0e-6,
        "matched_bound_initial_gradient_matches_unbounded": abs(
            bounded_matched["initial_amplitude_gradient_mean"]
            - candidate["initial_amplitude_gradient_mean"]
        )
        <= 1.0e-6,
        "added_parameters_under_one_percent": 0 < added < source_parameters * 0.01,
        "zero_control_has_no_frequency_information": control["descriptor_abs_sum"] == 0.0,
        "candidate_has_frequency_information": candidate["descriptor_abs_sum"] > 0.0,
        "bounded_has_frequency_information": bounded_invalid["descriptor_abs_sum"]
        > 0.0,
        "matched_bound_has_frequency_information": bounded_matched[
            "descriptor_abs_sum"
        ]
        > 0.0,
        "classification_path_only": all(
            record["gates"]["active_preserves_boxes"] for record in records.values()
        ),
        "test_accessed": False,
    }
    all_arm_gates = all(all(record["gates"].values()) for record in records.values())
    decision = "PASS" if all_arm_gates and all(
        value for key, value in global_gates.items() if key != "test_accessed"
    ) and not global_gates["test_accessed"] else "FAIL"
    result = {
        "format": "coffee_detector.af2_ffa.static_audit.v1",
        "decision": decision,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_parameters": source_parameters,
        "candidate_parameters": candidate["parameters"],
        "added_parameters": added,
        "added_fraction": added / source_parameters,
        "records": records,
        "gates": global_gates,
        "training_authorized": decision == "PASS",
        "test_access_authorized": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Static audit AF2 feature-frequency adapter")
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    result = run_af2_ffa_static_audit(
        args.af2_checkpoint,
        args.output,
        device=args.device,
        image_size=args.image_size,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
