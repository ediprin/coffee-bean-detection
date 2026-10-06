"""One-shot validation screen for the frozen train-only DVSR selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coffee_detector.analysis.defectoscafeverde_dual_side_routability import (
    FORMAT as AUDIT_FORMAT,
    _evaluate_choice,
    _selector_view,
    view_features,
)
from coffee_detector.experiments.run_defectoscafeverde_dual_view_fusion import (
    _json,
    _metrics,
    _sha256,
)
from coffee_detector.experiments.run_defectoscafeverde_dvf1_confirmation import (
    FORMAT as ARM_FORMAT,
    METRICS,
)
from coffee_detector.experiments.run_defectoscafeverde_dvf1_confirmation_decision import (
    FORMAT as DECISION_FORMAT,
)


FORMAT = "coffee_detector.defectoscafeverde.dvsr1.validation_screen.v1"
PROTOCOL = "defectoscafeverde-dvsr1-validation-screen-v1"


def _endpoint_arrays(cache: dict, model: dict) -> tuple[dict, dict]:
    logits = cache["view_logits"].float()
    labels = cache["labels"].long()
    ious = cache["ious"].float()
    scores = logits.sigmoid()
    confidence, predicted_class = scores.max(dim=2)
    paper_view = confidence.argmax(dim=1).numpy()
    features = view_features(logits.reshape(-1, logits.shape[-1])).reshape(
        len(logits), 2, -1
    )
    difference = (features[:, 0] - features[:, 1]).numpy().astype(np.float64)
    selector_view = _selector_view(difference, paper_view, model)
    correct = ((predicted_class == labels[:, None]) & (ious >= 0.5)).numpy()
    paper_correct = _evaluate_choice(paper_view, correct)
    selector_correct = _evaluate_choice(selector_view, correct)
    oracle_correct = correct.any(axis=1)
    arrays = {
        "paper_view": paper_view,
        "selector_view": selector_view,
        "paper_correct": paper_correct,
        "selector_correct": selector_correct,
        "oracle_correct": oracle_correct,
    }
    names = {int(key): str(value) for key, value in cache["names"].items()}
    values = {
        "PAPER_MAX_CONFIDENCE": _metrics(
            torch.from_numpy(paper_correct), labels, names
        ),
        "DVSR1": _metrics(torch.from_numpy(selector_correct), labels, names),
        "PAIR_ORACLE": _metrics(torch.from_numpy(oracle_correct), labels, names),
    }
    return values, arrays


def run_dvsr1_screen(
    train_audit: str | Path,
    validation_cache: str | Path,
    confirmation_arm: str | Path,
    confirmation_decision: str | Path,
    output: str | Path,
) -> dict:
    audit_path = Path(train_audit).expanduser().resolve()
    cache_path = Path(validation_cache).expanduser().resolve()
    arm_path = Path(confirmation_arm).expanduser().resolve()
    decision_path = Path(confirmation_decision).expanduser().resolve()
    audit = _json(audit_path, "DVSR train audit")
    arm = _json(arm_path, "DVF1 confirmation arm")
    decision = _json(decision_path, "DVF1 confirmation decision")
    if (
        audit.get("format") != AUDIT_FORMAT
        or audit.get("decision") != "AUTHORIZE_DVSR1_VALIDATION_SCREEN"
        or audit.get("validation_images_accessed") is not False
        or audit.get("test_images_accessed") is not False
    ):
        raise RuntimeError("DVSR train audit tidak mengotorisasi DVSR1")
    if (
        arm.get("format") != ARM_FORMAT
        or arm.get("status") != "COMPLETE"
        or arm.get("test_images_accessed") is not False
    ):
        raise RuntimeError("DVF1 confirmation arm tidak valid")
    if (
        decision.get("format") != DECISION_FORMAT
        or decision.get("decision") != "PASS"
        or decision.get("next") != "PROMOTE_DVF1_DUAL_VIEW_FUSION"
        or decision.get("training_executed") is not False
        or decision.get("test_images_accessed") is not False
    ):
        raise RuntimeError("DVF1 confirmation decision tidak valid")
    seed = str(arm["seed"])
    hashes = audit.get("contract", {})
    if hashes.get("confirmation_arm_sha256") != _sha256(arm_path):
        raise RuntimeError("Confirmation arm berbeda dari DVSR audit")
    if hashes.get("confirmation_decision_sha256") != _sha256(decision_path):
        raise RuntimeError("Confirmation decision berbeda dari DVSR audit")
    if decision["artifacts"][seed]["sha256"] != _sha256(arm_path):
        raise RuntimeError("Confirmation arm berbeda dari aggregate decision")
    if not cache_path.is_file() or arm["run_contract"]["validation_cache_sha256"] != _sha256(cache_path):
        raise RuntimeError("Validation cache berbeda dari confirmation contract")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("split") != "val" or "test" in cache_path.name.lower():
        raise RuntimeError("DVSR1 hanya menerima validation cache")
    selector = audit["final_train_selector"]
    model = {
        "scale": np.asarray(selector["scale"], dtype=np.float64),
        "weights": np.asarray(selector["weights"], dtype=np.float64),
    }
    values, arrays = _endpoint_arrays(cache, model)
    confirmation_values = arm["values"]
    calibration = {
        endpoint: {
            metric: values[endpoint][metric] - confirmation_values[endpoint][metric]
            for metric in METRICS
        }
        for endpoint in ("PAPER_MAX_CONFIDENCE", "PAIR_ORACLE")
    }
    dvf1_mean = {
        metric: decision["aggregate"][metric]["dvf1_mean"] for metric in METRICS
    }
    vs_paper = {
        metric: values["DVSR1"][metric] - values["PAPER_MAX_CONFIDENCE"][metric]
        for metric in METRICS
    }
    vs_dvf1 = {
        metric: values["DVSR1"][metric] - dvf1_mean[metric] for metric in METRICS
    }
    paper = arrays["paper_correct"]
    selector_correct = arrays["selector_correct"]
    transitions = {
        "rescues": int(((~paper) & selector_correct).sum()),
        "regressions": int((paper & (~selector_correct)).sum()),
        "both_correct": int((paper & selector_correct).sum()),
        "both_wrong": int(((~paper) & (~selector_correct)).sum()),
    }
    paper_controls = {
        "physical_gain_over_paper_at_least_0_5_point": vs_paper["physical_pair_accuracy"] >= 0.005,
        "macro_not_lower_than_paper": vs_paper["macro_class_accuracy"] >= 0.0,
        "bottom3_not_lower_than_paper": vs_paper["bottom3_class_accuracy"] >= 0.0,
        "worst_drop_vs_paper_no_more_than_2_points": vs_paper["worst_class_accuracy"] >= -0.02,
    }
    overall_route = {
        "physical_gain_over_dvf1_mean_at_least_0_5_point": vs_dvf1["physical_pair_accuracy"] >= 0.005,
        "macro_not_lower_than_dvf1_mean": vs_dvf1["macro_class_accuracy"] >= 0.0,
        "bottom3_not_lower_than_dvf1_mean": vs_dvf1["bottom3_class_accuracy"] >= 0.0,
        "worst_drop_vs_dvf1_mean_no_more_than_2_points": vs_dvf1["worst_class_accuracy"] >= -0.02,
    }
    lower_tail_route = {
        "physical_within_0_2_point_of_dvf1_mean": vs_dvf1["physical_pair_accuracy"] >= -0.002,
        "macro_not_lower_than_dvf1_mean": vs_dvf1["macro_class_accuracy"] >= 0.0,
        "bottom3_gain_over_dvf1_mean_at_least_1_point": vs_dvf1["bottom3_class_accuracy"] >= 0.01,
        "worst_not_lower_than_dvf1_mean": vs_dvf1["worst_class_accuracy"] >= 0.0,
    }
    gates = {
        "endpoint_calibration_exact": all(
            value == 0.0 for row in calibration.values() for value in row.values()
        ),
        "all_12_validation_classes_present": all(
            not row["classes_without_ground_truth"] for row in values.values()
        ),
        "selector_fitting_not_executed": True,
        "detector_or_fuser_training_not_executed": True,
        "test_not_opened": True,
    }
    passed = (
        all(paper_controls.values())
        and (all(overall_route.values()) or all(lower_tail_route.values()))
        and all(gates.values())
    )
    result = {
        "format": FORMAT,
        "protocol": PROTOCOL,
        "values": values,
        "confirmed_dvf1_mean": dvf1_mean,
        "deltas_vs_paper": vs_paper,
        "deltas_vs_confirmed_dvf1_mean": vs_dvf1,
        "transitions_vs_paper": transitions,
        "endpoint_calibration": calibration,
        "criteria": {
            "paper_controls": paper_controls,
            "overall_route": overall_route,
            "lower_tail_route": lower_tail_route,
        },
        "gates": gates,
        "decision": "PASS" if passed else "FAIL",
        "next": "PROMOTE_DVSR1_SIDE_RELIABILITY" if passed else "RETAIN_CONFIRMED_DVF1",
        "claim_boundary": "single frozen validation screen; no locked-test claim",
        "selector_fitting_executed": False,
        "detector_training_executed": False,
        "fuser_training_executed": False,
        "validation_images_accessed": True,
        "test_images_accessed": False,
        "contract": {
            "train_audit_sha256": _sha256(audit_path),
            "validation_cache_sha256": _sha256(cache_path),
            "confirmation_arm_sha256": _sha256(arm_path),
            "confirmation_decision_sha256": _sha256(decision_path),
        },
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen DVSR1 validation screen")
    parser.add_argument("--train-audit", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--confirmation-arm", required=True)
    parser.add_argument("--confirmation-decision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_dvsr1_screen(
        args.train_audit,
        args.validation_cache,
        args.confirmation_arm,
        args.confirmation_decision,
        args.output,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
