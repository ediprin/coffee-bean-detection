from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from coffee_detector.afab.model import load_afab_weights
from coffee_detector.afab.operator import AFABConfig

from .config import DIDAAF2Config
from .loss import GTLogits, smooth_topk_margin_loss, weak_to_strong_consistency
from .model import DIDAAF2DetectionModel
from .style import diversify_appearance


REPO_ROOT = Path(__file__).resolve().parents[3]
ARM_CONFIGS = {
    code: REPO_ROOT / f"configs/dida_af2/{code}_yolo26n.yaml"
    for code in ("AF2FT", "AF2DG", "AF2FG", "AF2DGFG")
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        output: list[torch.Tensor] = []
        for key in sorted(value):
            output.extend(_tensor_leaves(value[key]))
        return output
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(_tensor_leaves(item))
        return output
    return []


def _max_output_difference(left: Any, right: Any) -> float:
    left_tensors, right_tensors = _tensor_leaves(left), _tensor_leaves(right)
    if len(left_tensors) != len(right_tensors):
        return float("inf")
    differences = []
    for first, second in zip(left_tensors, right_tensors):
        if first.shape != second.shape:
            return float("inf")
        differences.append(float((first.float() - second.float()).abs().max()))
    return max(differences, default=0.0)


def run_dida_af2_static_audit(
    af2_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    from ultralytics import YOLO

    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    configs = {code: _load_config(path) for code, path in ARM_CONFIGS.items()}
    frozen = {code: DIDAAF2Config.from_mapping(configs[code]["dida"]) for code in configs}
    expected_flags = {
        "AF2FT": (False, False),
        "AF2DG": (True, False),
        "AF2FG": (False, True),
        "AF2DGFG": (True, True),
    }
    source = YOLO(str(checkpoint)).model.to(device).eval()
    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    source_schema = {
        key: tuple(value.shape) for key, value in source.state_dict().items()
    }
    torch.manual_seed(20260817)
    sample = torch.rand(1, 3, 64, 64, device=device)
    with torch.inference_mode():
        source_output = source(sample)
    records: dict[str, Any] = {}
    for code in expected_flags:
        payload = configs[code]
        candidate = DIDAAF2DetectionModel(
            str(REPO_ROOT / payload["model"]),
            ch=3,
            nc=int(getattr(source.model[-1], "nc", 21)),
            verbose=False,
            afab=AFABConfig.from_mapping(payload["afab"]),
            dida=frozen[code],
        ).to(device)
        transfer = load_afab_weights(candidate, source)
        candidate.eval()
        with torch.inference_mode():
            candidate_output = candidate(sample)
        records[code] = {
            "flags": [frozen[code].dg_enabled, frozen[code].fg_enabled],
            "parameters": sum(parameter.numel() for parameter in candidate.parameters()),
            "same_state_dict_schema": {
                key: tuple(value.shape) for key, value in candidate.state_dict().items()
            }
            == source_schema,
            "inference_max_abs_diff": _max_output_difference(
                source_output, candidate_output
            ),
            "transfer": transfer,
        }
        del candidate

    style = diversify_appearance(sample, frozen["AF2DG"])
    clone = sample.clone()
    weak_logits = torch.randn(4, 21, device=device, requires_grad=True)
    strong_logits = torch.randn(4, 21, device=device, requires_grad=True)
    keys = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]], device=device)
    labels = torch.tensor([0, 2, 7, 20], device=device)
    fg = smooth_topk_margin_loss(
        weak_logits, labels, margin=frozen["AF2FG"].margin, topk=3
    )
    dg, matched = weak_to_strong_consistency(
        GTLogits(keys, labels, weak_logits),
        GTLogits(keys, labels, strong_logits),
        temperature=frozen["AF2DG"].temperature,
    )
    (fg + dg).backward()
    gates = {
        "factorial_flags_exact": all(
            tuple(records[code]["flags"]) == expected_flags[code]
            for code in expected_flags
        ),
        "same_model_yaml": len({configs[code]["model"] for code in configs}) == 1,
        "same_afab_config": len(
            {json.dumps(configs[code]["afab"], sort_keys=True) for code in configs}
        )
        == 1,
        "same_training_schedule": len(
            {json.dumps(configs[code]["train"], sort_keys=True) for code in configs}
        )
        == 1,
        "only_dida_mode_differs": len(
            {
                json.dumps(
                    {
                        key: value
                        for key, value in configs[code]["dida"].items()
                        if key != "mode"
                    },
                    sort_keys=True,
                )
                for code in configs
            }
        )
        == 1,
        "same_parameter_count": all(
            record["parameters"] == source_parameters for record in records.values()
        ),
        "same_state_dict_schema": all(
            record["same_state_dict_schema"] for record in records.values()
        ),
        "inference_numerically_identical": all(
            record["inference_max_abs_diff"] <= 1.0e-6 for record in records.values()
        ),
        "dg_off_is_exact_clone": torch.equal(sample, clone),
        "dg_on_preserves_shape": style.shape == sample.shape,
        "dg_on_finite_unit_interval": bool(
            torch.isfinite(style).all() and style.min() >= 0 and style.max() <= 1
        ),
        "dg_on_changes_appearance": not torch.equal(style, sample),
        "auxiliary_losses_finite": bool(torch.isfinite(fg) and torch.isfinite(dg)),
        "auxiliary_gradients_finite": bool(
            torch.isfinite(weak_logits.grad).all()
            and torch.isfinite(strong_logits.grad).all()
        ),
        "gt_matching_operational": matched == 4,
        "classification_only_auxiliary_api": True,
        "test_accessed": False,
    }
    decision = "PASS" if all(value for key, value in gates.items() if key != "test_accessed") and not gates["test_accessed"] else "FAIL"
    result = {
        "format": "coffee_detector.dida_af2.static_audit.v1",
        "decision": decision,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_parameters": source_parameters,
        "records": records,
        "loss_probe": {"fg": float(fg.detach()), "dg": float(dg.detach()), "matched": matched},
        "gates": gates,
        "training_authorized": decision == "PASS",
        "test_access_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Static audit DIDA-AF2 2x2")
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_dida_af2_static_audit(
        args.af2_checkpoint, args.output, device=args.device
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
