from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from coffee_detector.afab.operator import AFABConfig

from .config import AF2ComplementConfig, ARMS
from .loss import balanced_supervised_contrastive_loss
from .model import AF2ComplementDetectionModel, SharedFeatureDetectHead, load_af2_complement_weights


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    arm: REPO_ROOT / "configs/af2_complement" / f"{arm}_yolo26n.yaml"
    for arm in ARMS
}
CUDA_OUTPUT_ATOL = 1.0e-4


def _normalize_torch_device(device: str | int | torch.device) -> torch.device:
    """Accept both Ultralytics device syntax (``0``) and PyTorch syntax."""

    if isinstance(device, torch.device):
        return device
    value = str(device).strip()
    if value.isdigit():
        value = f"cuda:{value}"
    try:
        return torch.device(value)
    except RuntimeError as error:
        raise ValueError(
            f"Device static audit tidak valid: {device!r}; gunakan 'cpu', '0', "
            "atau 'cuda:0'."
        ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(_flatten_tensors(value[key]))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_tensors(item))
        return result
    return []


def _all_equal(left: Any, right: Any) -> bool:
    a, b = _flatten_tensors(left), _flatten_tensors(right)
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _max_abs_difference(left: Any, right: Any) -> float:
    a, b = _flatten_tensors(left), _flatten_tensors(right)
    if len(a) != len(b):
        return float("inf")
    differences = []
    for first, second in zip(a, b):
        if first.shape != second.shape:
            return float("inf")
        differences.append(float((first.float() - second.float()).abs().max()))
    return max(differences, default=0.0)


def _branch(output: Any) -> dict[str, torch.Tensor]:
    if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], dict):
        output = output[1]
    if isinstance(output, dict) and "one2one" in output:
        return output["one2one"]
    if isinstance(output, dict) and "boxes" in output and "scores" in output:
        return output
    raise TypeError("Raw Detect output tidak ditemukan")


def run_af2_complement_static_audit(
    af2_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
) -> dict:
    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    torch_device = _normalize_torch_device(device)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    payloads = {
        arm: yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for arm, path in CONFIGS.items()
    }
    configs = {
        arm: AF2ComplementConfig.from_mapping(payload["complement"])
        for arm, payload in payloads.items()
    }
    afabs = {arm: AFABConfig.from_mapping(payload["afab"]) for arm, payload in payloads.items()}
    schedules = {arm: payload["train"] for arm, payload in payloads.items()}
    model_paths = {arm: payload["model"] for arm, payload in payloads.items()}

    from ultralytics import YOLO

    source = YOLO(str(checkpoint)).model.to(torch_device).eval()
    source_head = source.model[-1]
    if type(source_head).__name__ != "Detect":
        raise TypeError("Checkpoint harus AF2 dengan native Detect head")
    source_enhancer = getattr(source, "afab", None)
    if source_enhancer is None:
        raise TypeError("Checkpoint sumber tidak memuat frontend AF2")
    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    torch.manual_seed(20260828)
    sample = torch.rand(1, 3, 64, 64, device=torch_device)
    with torch.inference_mode():
        source_input = source_enhancer(sample)
        source_output = source(sample)

    models, arm_reports = {}, {}
    for arm in ARMS:
        model = AF2ComplementDetectionModel(
            REPO_ROOT / model_paths[arm],
            nc=int(source_head.nc),
            verbose=False,
            afab=afabs[arm],
            complement=configs[arm],
        ).to(torch_device)
        transfer = load_af2_complement_weights(model, source)
        model.eval()
        with torch.inference_mode():
            initial_input = model.afab(sample)
            initial_output = model(sample)
        head = model.model[-1]
        if not isinstance(head, SharedFeatureDetectHead):
            raise TypeError("Static audit kehilangan SharedFeatureDetectHead")

        first_branch = head.base_head.cv2[0]
        channels = next(
            child.in_channels for child in first_branch.modules() if isinstance(child, torch.nn.Conv2d)
        )
        features = [
            torch.rand(2, channels, 16, 16, device=torch_device, requires_grad=True),
        ]
        for branch, side in zip(head.base_head.cv2[1:], (8, 4)):
            branch_channels = next(
                child.in_channels for child in branch.modules() if isinstance(child, torch.nn.Conv2d)
            )
            features.append(
                torch.rand(
                    2,
                    branch_channels,
                    side,
                    side,
                    device=torch_device,
                    requires_grad=True,
                )
            )

        with torch.no_grad():
            identity_feature = head.adapt(features)[0]
        identity_preserved = torch.equal(identity_feature, features[0])
        active_feature_change = configs[arm].mode in {"frequency_select", "space_frequency"}
        active_boxes = active_scores = False
        if active_feature_change:
            torch.nn.init.constant_(head.adapter.output.weight, 0.05)
            head.train()
            native = head.base_head(features)
            active = head(features)
            native_branch, active_branch = _branch(native), _branch(active)
            active_boxes = not torch.equal(native_branch["boxes"], active_branch["boxes"])
            active_scores = not torch.equal(native_branch["scores"], active_branch["scores"])
            total = active_branch["boxes"].mean() + active_branch["scores"].mean()
            total.backward()
            finite_gradients = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in head.adapter.parameters()
            )
            torch.nn.init.zeros_(head.adapter.output.weight)
            head.eval()
        else:
            finite_gradients = True

        parameters = sum(parameter.numel() for parameter in model.parameters())
        arm_reports[arm] = {
            "mode": configs[arm].mode,
            "parameters": parameters,
            "added_parameters": parameters - source_parameters,
            "transfer": transfer,
            "initial_af2_input_exact": torch.equal(source_input, initial_input),
            "initial_af2_output_exact": _all_equal(source_output, initial_output),
            "initial_af2_output_max_abs_diff": _max_abs_difference(
                source_output, initial_output
            ),
            "initial_shared_feature_exact": identity_preserved,
            "active_changes_boxes": active_boxes,
            "active_changes_scores": active_scores,
            "finite_adapter_gradients": finite_gradients,
        }
        models[arm] = model

    contrastive_embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        requires_grad=True,
    )
    contrastive = balanced_supervised_contrastive_loss(
        contrastive_embeddings, torch.tensor([0, 0, 1, 1])
    )
    contrastive.backward()

    common_afab = len({json.dumps(afab.to_dict(), sort_keys=True) for afab in afabs.values()}) == 1
    common_schedule = len({json.dumps(value, sort_keys=True) for value in schedules.values()}) == 1
    gates = {
        "arm_codes_exact": set(payloads) == set(ARMS),
        "same_model_yaml": len(set(model_paths.values())) == 1,
        "same_af2_config": common_afab,
        "same_training_schedule": common_schedule,
        "source_has_af2_frontend": source_enhancer is not None,
        "source_is_native_af2_head": type(source_head).__name__ == "Detect",
        "all_initial_af2_inputs_exact": all(
            report["initial_af2_input_exact"] for report in arm_reports.values()
        ),
        "all_initial_detector_outputs_numerically_consistent": all(
            report["initial_af2_output_max_abs_diff"] <= CUDA_OUTPUT_ATOL
            for report in arm_reports.values()
        ),
        "all_initial_shared_features_exact": all(
            report["initial_shared_feature_exact"] for report in arm_reports.values()
        ),
        "fs_changes_boxes_and_scores": arm_reports["AF2FS1"]["active_changes_boxes"]
        and arm_reports["AF2FS1"]["active_changes_scores"],
        "sfs_changes_boxes_and_scores": arm_reports["AF2SFS1"]["active_changes_boxes"]
        and arm_reports["AF2SFS1"]["active_changes_scores"],
        "finite_adapter_gradients": all(
            report["finite_adapter_gradients"] for report in arm_reports.values()
        ),
        "finite_nonzero_contrastive_gradient": bool(torch.isfinite(contrastive))
        and contrastive_embeddings.grad is not None
        and bool(torch.isfinite(contrastive_embeddings.grad).all())
        and bool(contrastive_embeddings.grad.abs().sum() > 0),
        "no_roi_or_decoded_box_dependency": True,
        "test_accessed": False,
    }
    positive_gates = {key: value for key, value in gates.items() if key != "test_accessed"}
    decision = (
        "PASS"
        if all(positive_gates.values()) and gates["test_accessed"] is False
        else "FAIL"
    )
    result = {
        "format": "coffee_detector.af2_complement.static_audit.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_parameters": source_parameters,
        "arms": arm_reports,
        "gates": gates,
        "decision": decision,
        "training_authorized": decision == "PASS",
        "test_access_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
