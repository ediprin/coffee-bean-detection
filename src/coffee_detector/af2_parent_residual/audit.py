"""Static identity, isolation, and matched-control audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from coffee_detector.afab import AFABConfig

from .config import AF2ParentResidualConfig
from .model import (
    AF2ParentResidualDetectionModel,
    freeze_for_parent_residual,
    load_af2_parent_residual_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ARMS = ("AF2SAF0", "AF2SAF1", "AF2IGEM0", "AF2IGEM1")
CONFIGS = {code: REPO_ROOT / f"configs/af2_parent_residual/{code}.yaml" for code in ARMS}
ATOL = 5.0e-5
RTOL = 1.0e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in module.state_dict().items()}


def _activate_last_projection(model: AF2ParentResidualDetectionModel) -> None:
    head = model.model[-1]
    with torch.no_grad():
        if head.config.family == "saf":
            for layer in head.residual.class_corrections:
                layer.weight.fill_(1.0)
        else:
            for level in head.residual:
                level.class_correction.weight.fill_(1.0)


def run_af2_parent_residual_static_audit(
    af2_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 64,
) -> dict[str, Any]:
    from ultralytics import YOLO

    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payloads = {
        code: yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for code, path in CONFIGS.items()
    }
    source = YOLO(str(checkpoint)).model.to(device).eval()
    if getattr(source, "afab", None) is None:
        raise RuntimeError("Checkpoint sumber bukan AF2")
    torch.manual_seed(20260823)
    image = torch.rand(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        native = source(image)
    records: dict[str, Any] = {}
    models = {}
    for code, payload in payloads.items():
        config = AF2ParentResidualConfig.from_mapping(payload["parent_residual"])
        model = AF2ParentResidualDetectionModel(
            str(REPO_ROOT / payload["model"]),
            nc=int(source.model[-1].nc),
            verbose=False,
            afab=AFABConfig.from_mapping(payload["afab"]),
            parent_residual=config,
        ).to(device)
        transfer = load_af2_parent_residual_weights(model, source)
        model.eval()
        with torch.inference_mode():
            identity = model(image)
        boxes = identity[1]["one2one"]["boxes"]
        scores = identity[1]["one2one"]["scores"]
        native_boxes = native[1]["one2one"]["boxes"]
        native_scores = native[1]["one2one"]["scores"]
        identity_close = torch.allclose(boxes, native_boxes, atol=ATOL, rtol=RTOL) and torch.allclose(
            scores, native_scores, atol=ATOL, rtol=RTOL
        )
        before_boxes, before_scores = boxes.clone(), scores.clone()
        _activate_last_projection(model)
        with torch.inference_mode():
            active = model(image)
        active_boxes = active[1]["one2one"]["boxes"]
        active_scores = active[1]["one2one"]["scores"]
        freeze = freeze_for_parent_residual(model)
        model.train(True)
        model.zero_grad(set_to_none=True)
        training_output = model(image)["one2many"]
        objective = training_output["scores"].square().mean()
        if config.family == "igem":
            objective = objective + sum(
                value.square().mean()
                for value in training_output["parent_residual_mask_logits"]
            )
        objective.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
        records[code] = {
            "family": config.family,
            "conditioning": config.conditioning,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": freeze["trainable"],
            "state_schema": _schema(model),
            "transfer": transfer,
            "initial_af2_numerically_preserved": identity_close,
            "active_boxes_bitwise_preserved": torch.equal(before_boxes, active_boxes),
            "active_score_max_abs_diff": float((before_scores - active_scores).abs().max()),
            "finite_nonzero_residual_gradients": all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in trainable
            ) and any(bool(parameter.grad.abs().max() > 0) for parameter in trainable),
            "frozen_parent_has_no_gradients": all(parameter.grad is None for parameter in frozen),
        }
        models[code] = model

    gates: dict[str, bool] = {"source_is_af2": True, "test_accessed": False}
    for family, control_code, candidate_code in (
        ("saf", "AF2SAF0", "AF2SAF1"),
        ("igem", "AF2IGEM0", "AF2IGEM1"),
    ):
        control, candidate = records[control_code], records[candidate_code]
        control_payload, candidate_payload = payloads[control_code], payloads[candidate_code]
        prefix = f"{family}_"
        gates.update(
            {
                prefix + "same_model_yaml": control_payload["model"] == candidate_payload["model"],
                prefix + "same_af2_config": control_payload["afab"] == candidate_payload["afab"],
                prefix + "same_training_schedule": control_payload["train"] == candidate_payload["train"],
                prefix + "same_parameter_count": control["parameters"] == candidate["parameters"],
                prefix + "same_trainable_count": control["trainable_parameters"] == candidate["trainable_parameters"],
                prefix + "same_state_schema": control["state_schema"] == candidate["state_schema"],
                prefix + "only_conditioning_differs": {
                    key: value
                    for key, value in control_payload["parent_residual"].items()
                    if key != "conditioning"
                } == {
                    key: value
                    for key, value in candidate_payload["parent_residual"].items()
                    if key != "conditioning"
                },
                prefix + "control_receives_zero_information": control["conditioning"] == "zero",
                prefix + "candidate_receives_features": candidate["conditioning"] == "feature",
            }
        )
    for code, record in records.items():
        gates[f"{code}_initial_identity"] = record["initial_af2_numerically_preserved"]
        gates[f"{code}_boxes_preserved"] = record["active_boxes_bitwise_preserved"]
        gates[f"{code}_finite_gradients"] = record["finite_nonzero_residual_gradients"]
        gates[f"{code}_frozen_parent"] = record["frozen_parent_has_no_gradients"]
        if record["conditioning"] == "feature":
            gates[f"{code}_active_scores"] = record["active_score_max_abs_diff"] > 0.0
        else:
            gates[f"{code}_zero_information_identity"] = record["active_score_max_abs_diff"] == 0.0
    decision = "PASS" if all(value for key, value in gates.items() if key != "test_accessed") and not gates["test_accessed"] else "FAIL"
    result = {
        "format": "coffee_detector.af2_parent_residual.static_audit.v1",
        "decision": decision,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "records": records,
        "gates": gates,
        "training_authorized": decision == "PASS",
        "test_access_authorized": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Static audit AF2 parent residual")
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()
    result = run_af2_parent_residual_static_audit(
        args.af2_checkpoint,
        args.output,
        device=args.device,
        image_size=args.image_size,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
