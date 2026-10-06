"""Final one-shot locked-test evaluation for promoted DVSR1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coffee_detector.analysis.defectoscafeverde_dual_side_routability import (
    FORMAT as AUDIT_FORMAT,
)
from coffee_detector.defectos_dual_view.model import (
    DualViewFusionConfig,
    SymmetricDualViewFuser,
)
from coffee_detector.experiments.run_defectoscafeverde_dual_view_fusion import (
    D0_PROTOCOL,
    FORMAT as SEED42_FORMAT,
    _device,
    _json,
    _metrics,
    _sha256,
    build_pair_cache,
)
from coffee_detector.experiments.run_defectoscafeverde_dvf1_confirmation import (
    FORMAT as ARM_FORMAT,
    METRICS,
)
from coffee_detector.experiments.run_defectoscafeverde_dvf1_confirmation_decision import (
    FORMAT as DECISION_FORMAT,
)
from coffee_detector.experiments.run_defectoscafeverde_dvsr1_screen import (
    FORMAT as DVSR1_FORMAT,
    _endpoint_arrays,
)


FORMAT = "coffee_detector.defectoscafeverde.dvsr1.locked_test.v1"
PROTOCOL = "defectoscafeverde-dvsr1-locked-test-v1"


def _load_fuser(path: Path, expected_sha: str, device: torch.device) -> SymmetricDualViewFuser:
    if not path.is_file() or _sha256(path) != expected_sha:
        raise RuntimeError(f"Checkpoint fuser tidak cocok: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model = SymmetricDualViewFuser(
        DualViewFusionConfig.from_mapping(state["config"])
    ).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def _fuser_correct(
    fuser: SymmetricDualViewFuser, cache: dict, device: torch.device
) -> torch.Tensor:
    logits = cache["view_logits"].to(device)
    scores = logits.sigmoid()
    labels = cache["labels"].to(device)
    ious = cache["ious"].to(device)
    with torch.inference_mode():
        fused_class = fuser(logits).argmax(dim=1)
    rows = torch.arange(len(labels), device=device)
    class_scores = scores.gather(
        2, fused_class[:, None, None].expand(-1, 2, 1)
    ).squeeze(2)
    selected_view = class_scores.argmax(dim=1)
    return (fused_class == labels) & (ious[rows, selected_view] >= 0.5)


def run_locked_test(
    data_root: str | Path,
    grouped_manifest: str | Path,
    grouped_audit: str | Path,
    d0_result: str | Path,
    d0_checkpoint: str | Path,
    seed42_result: str | Path,
    seed123_result: str | Path,
    seed2026_result: str | Path,
    confirmation_decision: str | Path,
    train_audit: str | Path,
    dvsr1_result: str | Path,
    output_root: str | Path,
    *,
    device: str = "cpu",
    authorize_test: bool = False,
) -> dict:
    if not authorize_test:
        raise PermissionError("Locked test memerlukan --authorize-test")
    root = Path(data_root).expanduser().resolve()
    manifest_path = Path(grouped_manifest).expanduser().resolve()
    grouped_audit_path = Path(grouped_audit).expanduser().resolve()
    d0_result_path = Path(d0_result).expanduser().resolve()
    d0_checkpoint_path = Path(d0_checkpoint).expanduser().resolve()
    result_paths = {
        42: Path(seed42_result).expanduser().resolve(),
        123: Path(seed123_result).expanduser().resolve(),
        2026: Path(seed2026_result).expanduser().resolve(),
    }
    decision_path = Path(confirmation_decision).expanduser().resolve()
    train_audit_path = Path(train_audit).expanduser().resolve()
    dvsr1_path = Path(dvsr1_result).expanduser().resolve()
    destination_root = Path(output_root).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / "DVSR1_locked_test_result.json"

    d0 = _json(d0_result_path, "D0 result")
    rows = {seed: _json(path, f"DVF1 seed {seed}") for seed, path in result_paths.items()}
    decision = _json(decision_path, "DVF1 confirmation decision")
    audit = _json(train_audit_path, "DVSR train audit")
    validation = _json(dvsr1_path, "DVSR1 validation result")
    if (
        d0.get("protocol") != D0_PROTOCOL
        or d0.get("arm") != "D0DIRECT"
        or d0.get("seed") != 42
        or d0.get("test_images_accessed") is not False
        or _sha256(d0_checkpoint_path) != d0.get("checkpoint_sha256")
    ):
        raise RuntimeError("D0 contract tidak valid")
    if rows[42].get("format") != SEED42_FORMAT or rows[42].get("decision") != "PASS":
        raise RuntimeError("DVF1 seed 42 tidak valid")
    for seed in (123, 2026):
        if rows[seed].get("format") != ARM_FORMAT or rows[seed].get("status") != "COMPLETE":
            raise RuntimeError(f"DVF1 seed {seed} tidak valid")
    if (
        decision.get("format") != DECISION_FORMAT
        or decision.get("decision") != "PASS"
        or decision.get("next") != "PROMOTE_DVF1_DUAL_VIEW_FUSION"
    ):
        raise RuntimeError("DVF1 confirmation tidak valid")
    for seed, path in result_paths.items():
        if decision["artifacts"][str(seed)]["sha256"] != _sha256(path):
            raise RuntimeError(f"DVF1 seed {seed} berbeda dari confirmation")
    if (
        audit.get("format") != AUDIT_FORMAT
        or audit.get("decision") != "AUTHORIZE_DVSR1_VALIDATION_SCREEN"
        or validation.get("format") != DVSR1_FORMAT
        or validation.get("decision") != "PASS"
        or validation.get("next") != "PROMOTE_DVSR1_SIDE_RELIABILITY"
        or validation.get("test_images_accessed") is not False
    ):
        raise RuntimeError("DVSR1 belum dipromosikan secara valid")
    if validation["contract"]["train_audit_sha256"] != _sha256(train_audit_path):
        raise RuntimeError("DVSR train audit berbeda dari validation result")
    if validation["contract"]["confirmation_decision_sha256"] != _sha256(decision_path):
        raise RuntimeError("DVF1 confirmation berbeda dari validation result")
    seed42_contract = rows[42]["run_contract"]
    expected_dataset = {
        "grouped_manifest_sha256": _sha256(manifest_path),
        "grouped_audit_sha256": _sha256(grouped_audit_path),
        "d0_result_sha256": _sha256(d0_result_path),
        "d0_checkpoint_sha256": _sha256(d0_checkpoint_path),
    }
    if any(seed42_contract.get(key) != value for key, value in expected_dataset.items()):
        raise RuntimeError("Dataset atau D0 berbeda dari DVF1 contract")
    if (root / "train").exists() or (root / "val").exists() or not (root / "test").is_dir():
        raise RuntimeError("Runtime locked-test harus hanya memuat split test")

    output_contract = {
        "protocol": PROTOCOL,
        "grouped_manifest_sha256": _sha256(manifest_path),
        "grouped_audit_sha256": _sha256(grouped_audit_path),
        "d0_checkpoint_sha256": _sha256(d0_checkpoint_path),
        "dvsr1_validation_result_sha256": _sha256(dvsr1_path),
        "test_images_accessed": True,
    }
    if destination.is_file():
        cached = _json(destination, "Cached locked-test result")
        if cached.get("format") != FORMAT or cached.get("contract") != output_contract:
            raise RuntimeError("Cached locked-test result stale")
        cached["evaluation_executed_this_call"] = False
        return cached

    torch_device = _device(device)
    from ultralytics import YOLO

    network = YOLO(str(d0_checkpoint_path)).model.to(torch_device).eval()
    detector_config = {"image_size": 640, "max_det": 500}
    cache_path = destination_root / "test_pair_cache.pt"
    cache_contract = {
        **output_contract,
        **detector_config,
        "localization_box_source": "final_top_v1",
    }
    test_cache = build_pair_cache(
        network,
        root,
        manifest_path,
        "test",
        cache_path,
        cache_contract,
        image_size=detector_config["image_size"],
        max_det=detector_config["max_det"],
        device=torch_device,
    )
    selector = audit["final_train_selector"]
    selector_model = {
        "scale": np.asarray(selector["scale"], dtype=np.float64),
        "weights": np.asarray(selector["weights"], dtype=np.float64),
    }
    values, arrays = _endpoint_arrays(test_cache, selector_model)
    names = {int(key): str(value) for key, value in test_cache["names"].items()}
    labels = test_cache["labels"]
    checkpoints = {
        42: result_paths[42].parent / "DVF1_last.pt",
        123: result_paths[123].parent / "DVF1_seed123_last.pt",
        2026: result_paths[2026].parent / "DVF1_seed2026_last.pt",
    }
    dvf1_values = {}
    for seed, checkpoint in checkpoints.items():
        expected = rows[seed]["artifacts"]["fuser_checkpoint_sha256"]
        fuser = _load_fuser(checkpoint, expected, torch_device)
        correct = _fuser_correct(fuser, test_cache, torch_device).cpu()
        dvf1_values[str(seed)] = _metrics(correct, labels, names)
    dvf1_mean = {
        metric: float(np.mean([row[metric] for row in dvf1_values.values()]))
        for metric in METRICS
    }
    values["DVF1_BY_SEED"] = dvf1_values
    values["DVF1_MEAN"] = dvf1_mean
    vs_paper = {
        metric: values["DVSR1"][metric] - values["PAPER_MAX_CONFIDENCE"][metric]
        for metric in METRICS
    }
    vs_dvf1 = {
        metric: values["DVSR1"][metric] - dvf1_mean[metric] for metric in METRICS
    }
    paper_correct = arrays["paper_correct"]
    selector_correct = arrays["selector_correct"]
    transitions = {
        "rescues": int(((~paper_correct) & selector_correct).sum()),
        "regressions": int((paper_correct & (~selector_correct)).sum()),
        "both_correct": int((paper_correct & selector_correct).sum()),
        "both_wrong": int(((~paper_correct) & (~selector_correct)).sum()),
    }
    gates = {
        "dataset_and_checkpoint_contract_exact": True,
        "test_only_runtime": not (root / "train").exists() and not (root / "val").exists(),
        "all_12_test_classes_present": all(
            not values[key]["classes_without_ground_truth"]
            for key in ("PAPER_MAX_CONFIDENCE", "DVSR1", "PAIR_ORACLE")
        ),
        "detector_fuser_selector_training_not_executed": True,
        "validation_not_accessed": True,
        "test_access_authorized": True,
    }
    criteria = {
        "physical_improves_over_paper": vs_paper["physical_pair_accuracy"] > 0.0,
        "macro_not_lower_than_paper": vs_paper["macro_class_accuracy"] >= 0.0,
        "bottom3_not_lower_than_paper": vs_paper["bottom3_class_accuracy"] >= 0.0,
        "worst_drop_vs_paper_no_more_than_2_points": vs_paper["worst_class_accuracy"] >= -0.02,
        "physical_not_lower_than_dvf1_mean": vs_dvf1["physical_pair_accuracy"] >= 0.0,
        "macro_not_lower_than_dvf1_mean": vs_dvf1["macro_class_accuracy"] >= 0.0,
    }
    confirmed = all(gates.values()) and all(criteria.values())
    result = {
        "format": FORMAT,
        "protocol": PROTOCOL,
        "values": values,
        "deltas_vs_paper": vs_paper,
        "deltas_vs_dvf1_mean": vs_dvf1,
        "transitions_vs_paper": transitions,
        "criteria": criteria,
        "gates": gates,
        "decision": "FINAL_DVSR1_GENERALIZATION_CONFIRMED" if confirmed else "FINAL_DVSR1_GENERALIZATION_NOT_CONFIRMED",
        "next": "REPORT_LOCKED_TEST_AND_STOP",
        "test_pair_stats": test_cache["stats"],
        "contract": output_contract,
        "detector_training_executed": False,
        "fuser_training_executed": False,
        "selector_fitting_executed": False,
        "validation_images_accessed": False,
        "test_images_accessed": True,
        "evaluation_executed_this_call": True,
    }
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="DVSR1 final locked test")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-manifest", required=True)
    parser.add_argument("--grouped-audit", required=True)
    parser.add_argument("--d0-result", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--seed42-result", required=True)
    parser.add_argument("--seed123-result", required=True)
    parser.add_argument("--seed2026-result", required=True)
    parser.add_argument("--confirmation-decision", required=True)
    parser.add_argument("--train-audit", required=True)
    parser.add_argument("--dvsr1-result", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--authorize-test", action="store_true")
    args = parser.parse_args()
    result = run_locked_test(
        args.data_root,
        args.grouped_manifest,
        args.grouped_audit,
        args.d0_result,
        args.d0_checkpoint,
        args.seed42_result,
        args.seed123_result,
        args.seed2026_result,
        args.confirmation_decision,
        args.train_audit,
        args.dvsr1_result,
        args.output_root,
        device=args.device,
        authorize_test=args.authorize_test,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
