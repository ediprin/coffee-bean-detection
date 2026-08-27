"""Uniform weight soup of the three confirmed AF2 validation checkpoints.

This is a validation-only, zero-training study.  The coefficients are frozen
at one third before evaluation; validation is never used to choose weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn

from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import (
    load_faruq_grouped_summary,
)


SEEDS = (42, 123, 2026)
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)
PROTOCOL = "faruq-v3-af2-uniform-model-soup-validation-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> tuple[dict, nn.Module]:
    from ultralytics.utils.patches import torch_load

    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint bukan dictionary Ultralytics: {path}")
    model = payload.get("ema") or payload.get("model")
    if not isinstance(model, nn.Module):
        raise RuntimeError(f"Checkpoint tidak menyimpan model/EMA: {path}")
    return payload, model.float().cpu()


def _canonical_model_contract(model: nn.Module) -> dict:
    config = getattr(model, "afab_config", None)
    if config is None or getattr(config, "mode", None) != "af2":
        raise RuntimeError("Checkpoint bukan AF2 yang kompatibel")
    config_payload = config.to_dict() if hasattr(config, "to_dict") else vars(config)
    yaml_payload = getattr(model, "yaml", None)
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "afab_config": config_payload,
        "yaml": json.loads(json.dumps(yaml_payload, sort_keys=True, default=str)),
        "names": json.loads(json.dumps(getattr(model, "names", None), sort_keys=True, default=str)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def average_state_dicts(
    states: Iterable[Mapping[str, Tensor]],
) -> tuple[OrderedDict[str, Tensor], dict]:
    """Average floating tensors uniformly; require exact structural buffers."""

    frozen = list(states)
    if len(frozen) != 3:
        raise ValueError("Uniform AF2 soup memerlukan tepat tiga state dictionary")
    keys = list(frozen[0])
    if any(list(state) != keys for state in frozen[1:]):
        raise RuntimeError("State dictionary schema berbeda")

    output: OrderedDict[str, Tensor] = OrderedDict()
    floating, structural = 0, 0
    for key in keys:
        tensors = [state[key].detach().cpu() for state in frozen]
        reference = tensors[0]
        if any(value.shape != reference.shape for value in tensors[1:]):
            raise RuntimeError(f"Shape state berbeda: {key}")
        if any(value.dtype != reference.dtype for value in tensors[1:]):
            raise RuntimeError(f"Dtype state berbeda: {key}")
        if reference.is_floating_point() or reference.is_complex():
            accumulator_dtype = torch.complex128 if reference.is_complex() else torch.float64
            output[key] = torch.stack(
                [value.to(accumulator_dtype) for value in tensors]
            ).mean(dim=0).to(reference.dtype)
            floating += 1
        else:
            if any(not torch.equal(reference, value) for value in tensors[1:]):
                raise RuntimeError(f"Buffer struktural tidak identik: {key}")
            output[key] = reference.clone()
            structural += 1
    return output, {
        "state_tensors": len(keys),
        "averaged_floating_tensors": floating,
        "preserved_structural_tensors": structural,
    }


def _validate_confirmation(path: Path) -> tuple[dict, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("protocol")
        != "faruq-v3-af2-igem-paired-validation-confirmation-v1"
        or payload.get("seeds") != list(SEEDS)
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
        or payload.get("test_opened") is not False
        or payload.get("decisions", {}).get("AF2", {}).get("decision") != "PASS"
    ):
        raise RuntimeError("Konfirmasi AF2 bukan PASS validation-only tiga seed")
    reference = {
        metric: statistics.fmean(
            float(payload["per_seed"][str(seed)]["AF2"][metric]) for seed in SEEDS
        )
        for metric in METRICS
    }
    return payload, reference


def _decision(metrics: Mapping[str, float], reference: Mapping[str, float]) -> tuple[dict, str]:
    deltas = {metric: float(metrics[metric]) - float(reference[metric]) for metric in METRICS}
    criteria = {
        "macro_not_lower_than_af2_three_seed_mean": deltas["macro_map50_95"] >= 0.0,
        "bottom3_not_lower_than_af2_three_seed_mean": deltas["bottom3_class_map50_95"] >= 0.0,
        "worst_not_lower_than_af2_three_seed_mean": deltas["worst_class_map50_95"] >= 0.0,
        "tail_gain_at_least_0_5_point": max(
            deltas["bottom3_class_map50_95"], deltas["worst_class_map50_95"]
        ) >= 0.005,
    }
    return {"deltas": deltas, "criteria": criteria}, "RETAIN" if all(criteria.values()) else "REJECT"


def build_uniform_af2_soup(
    checkpoints: Iterable[str | Path], output_checkpoint: str | Path
) -> dict:
    paths = [Path(value).expanduser().resolve() for value in checkpoints]
    if len(paths) != 3 or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"Diperlukan tiga checkpoint AF2: {paths}")
    loaded = [_load_checkpoint(path) for path in paths]
    recorded_seeds = []
    for payload, path in ((item[0], path) for item, path in zip(loaded, paths)):
        train_args = payload.get("train_args")
        if not isinstance(train_args, dict) or "seed" not in train_args:
            raise RuntimeError(f"Checkpoint tidak merekam seed: {path}")
        recorded_seeds.append(int(train_args["seed"]))
    if tuple(recorded_seeds) != SEEDS:
        raise RuntimeError(f"Urutan seed checkpoint salah: {recorded_seeds} != {SEEDS}")
    contracts = [_canonical_model_contract(model) for _, model in loaded]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise RuntimeError("Arsitektur/config AF2 antar-seed tidak identik")

    states = [model.state_dict() for _, model in loaded]
    averaged, state_audit = average_state_dicts(states)
    soup_model = copy.deepcopy(loaded[0][1]).cpu()
    incompatible = soup_model.load_state_dict(averaged, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"State soup tidak strict: {incompatible}")

    output = Path(output_checkpoint).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = copy.copy(loaded[0][0])
    checkpoint.update(
        {
            "epoch": -1,
            "best_fitness": None,
            "model": soup_model.half(),
            "ema": None,
            "optimizer": None,
            "updates": None,
            "af2_uniform_soup": {
                "protocol": PROTOCOL,
                "seeds": list(SEEDS),
                "coefficients": [1.0 / 3.0] * 3,
                "source_sha256": [_sha256(path) for path in paths],
            },
        }
    )
    torch.save(checkpoint, output)
    reloaded_payload, reloaded_model = _load_checkpoint(output)
    reloaded_contract = _canonical_model_contract(reloaded_model)
    gates = {
        "three_unique_checkpoints": len(set(map(_sha256, paths))) == 3,
        "same_model_contract": all(contract == contracts[0] for contract in contracts),
        "same_parameter_count": len({contract["parameters"] for contract in contracts}) == 1,
        "strict_state_schema": True,
        "uniform_coefficients": True,
        "checkpoint_seeds_exact": tuple(recorded_seeds) == SEEDS,
        "reloaded_contract_identical": reloaded_contract == contracts[0],
        "checkpoint_has_no_optimizer": reloaded_payload.get("optimizer") is None,
        "no_trainable_parameters_added": reloaded_contract["parameters"] == contracts[0]["parameters"],
        "test_not_accessed": True,
    }
    return {
        "format": "coffee_detector.af2_uniform_soup.static_audit.v1",
        "protocol": PROTOCOL,
        "source_checkpoints": [str(path) for path in paths],
        "source_sha256": [_sha256(path) for path in paths],
        "seeds": list(SEEDS),
        "coefficients": [1.0 / 3.0] * 3,
        "model_contract": contracts[0],
        "state_audit": state_audit,
        "output_checkpoint": str(output),
        "output_sha256": _sha256(output),
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "training_executed": False,
        "test_images_accessed": False,
    }


def run_faruq_v3_af2_uniform_soup(
    data_root: str | Path,
    grouped_summary: str | Path,
    confirmation_summary: str | Path,
    checkpoints: Iterable[str | Path],
    output_root: str | Path,
    *,
    device: str | None = None,
) -> dict:
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    confirmation_path = Path(confirmation_summary).expanduser().resolve()
    _, reference = _validate_confirmation(confirmation_path)

    soup_checkpoint = output_root / "AF2_UNIFORM_SOUP/weights/best.pt"
    audit_path = output_root / "static_audit.json"
    audit = build_uniform_af2_soup(checkpoints, soup_checkpoint)
    audit["confirmation_summary"] = str(confirmation_path)
    audit["confirmation_sha256"] = _sha256(confirmation_path)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if audit["decision"] != "PASS":
        raise RuntimeError(f"Static soup gate gagal: {audit_path}")

    report_path = output_root / "val_reports/af2_uniform_soup_evaluation.json"
    report = evaluate(soup_checkpoint, data_root, report_path, split="val", device=device)
    metrics = {metric: float(report["metrics"][metric]) for metric in METRICS}
    missing = report["metrics"].get("classes_without_ground_truth", [])
    comparison, decision = _decision(metrics, reference)
    criteria = dict(comparison["criteria"])
    criteria["all_21_validation_classes_present"] = not missing
    criteria["test_not_opened"] = True
    if not all(criteria.values()):
        decision = "REJECT"
    result = {
        "format": "coffee_detector.af2_uniform_soup.result.v1",
        "protocol": PROTOCOL,
        "seeds": list(SEEDS),
        "coefficients": [1.0 / 3.0] * 3,
        "reference_af2_three_seed_mean": reference,
        "soup_metrics": metrics,
        "deltas": comparison["deltas"],
        "criteria": criteria,
        "decision": decision,
        "next": "RETAIN_AF2_UNIFORM_SOUP" if decision == "RETAIN" else "RETAIN_ORIGINAL_AF2",
        "checkpoint": str(soup_checkpoint),
        "checkpoint_sha256": _sha256(soup_checkpoint),
        "static_audit": str(audit_path),
        "evaluation_report": str(report_path),
        "training_executed": False,
        "test_images_accessed": False,
    }
    summary = output_root / "val_reports/af2_uniform_soup_decision.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniform AF2 model soup, validation only")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--confirmation-summary", required=True)
    parser.add_argument("--checkpoints", nargs=3, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = run_faruq_v3_af2_uniform_soup(
        args.data_root,
        args.grouped_summary,
        args.confirmation_summary,
        args.checkpoints,
        args.output_root,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
