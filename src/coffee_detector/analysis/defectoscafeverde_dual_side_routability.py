"""Train-only OOF audit for symmetric dual-view side reliability."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from coffee_detector.experiments.run_defectoscafeverde_dual_view_fusion import (
    _json,
    _sha256,
)
from coffee_detector.experiments.run_defectoscafeverde_dvf1_confirmation import (
    FORMAT as ARM_FORMAT,
)
from coffee_detector.experiments.run_defectoscafeverde_dvf1_confirmation_decision import (
    FORMAT as DECISION_FORMAT,
)


FORMAT = "coffee_detector.defectoscafeverde.dvsr_routability.v1"
FOLDS = 5
RIDGE_ALPHA = 10.0


def view_features(view_logits: torch.Tensor) -> torch.Tensor:
    """Return inference-visible per-view evidence for shared reliability scoring."""

    if view_logits.ndim != 2:
        raise ValueError("view_logits harus [N,C]")
    scores = view_logits.sigmoid()
    total = scores.sum(dim=1, keepdim=True).clamp_min(1e-8)
    distribution = scores / total
    top2 = scores.topk(2, dim=1)
    top_class = torch.nn.functional.one_hot(
        top2.indices[:, 0], num_classes=scores.shape[1]
    ).to(scores.dtype)
    entropy = -(distribution * distribution.clamp_min(1e-8).log()).sum(
        dim=1, keepdim=True
    )
    summaries = torch.cat(
        (
            top2.values[:, :1],
            top2.values[:, 1:2],
            top2.values[:, :1] - top2.values[:, 1:2],
            entropy,
            total,
        ),
        dim=1,
    )
    return torch.cat((scores, top_class, summaries), dim=1)


def _stable_fold(group_id: str, folds: int = FOLDS) -> int:
    digest = hashlib.sha256(str(group_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _fit_ridge_difference(
    difference: np.ndarray, targets: np.ndarray, alpha: float = RIDGE_ALPHA
) -> dict:
    if difference.ndim != 2 or len(difference) != len(targets):
        raise ValueError("Dimensi training selector tidak cocok")
    if len(difference) < 2 or not set(np.unique(targets)).issubset({-1.0, 1.0}):
        raise RuntimeError("Training selector tidak valid")
    # Sign augmentation makes the fit exactly antisymmetric without an
    # intercept or orientation-specific parameter.
    design = np.concatenate((difference, -difference), axis=0)
    labels = np.concatenate((targets, -targets), axis=0)
    scale = np.sqrt(np.mean(design * design, axis=0))
    scale[scale < 1e-8] = 1.0
    normalized = design / scale
    penalty = np.eye(normalized.shape[1], dtype=np.float64) * alpha
    weights = np.linalg.solve(
        normalized.T @ normalized + penalty, normalized.T @ labels
    )
    return {"scale": scale, "weights": weights}


def _selector_view(
    difference: np.ndarray,
    paper_view: np.ndarray,
    model: Mapping[str, np.ndarray],
) -> np.ndarray:
    score = (difference / model["scale"]) @ model["weights"]
    return np.where(score > 0.0, 0, np.where(score < 0.0, 1, paper_view))


def _correctness(cache: dict) -> dict:
    logits = cache["view_logits"].float()
    labels = cache["labels"].long()
    ious = cache["ious"].float()
    scores = logits.sigmoid()
    confidence, prediction = scores.max(dim=2)
    correct = (prediction == labels[:, None]) & (ious >= 0.5)
    paper_view = confidence.argmax(dim=1)
    rows = torch.arange(len(labels))
    paper_correct = correct[rows, paper_view]
    features = view_features(logits.reshape(-1, logits.shape[-1])).reshape(
        len(logits), 2, -1
    )
    probability = scores / scores.sum(dim=2, keepdim=True).clamp_min(1e-8)
    entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=2)
    top2 = scores.topk(2, dim=2).values
    margin = top2[:, :, 0] - top2[:, :, 1]
    margin_view = margin.argmax(dim=1)
    entropy_view = entropy.argmin(dim=1)
    return {
        "labels": labels.numpy(),
        "correct": correct.numpy(),
        "paper_view": paper_view.numpy(),
        "paper_correct": paper_correct.numpy(),
        "margin_view": margin_view.numpy(),
        "entropy_view": entropy_view.numpy(),
        "difference": (features[:, 0] - features[:, 1]).numpy().astype(np.float64),
    }


def _evaluate_choice(choice: np.ndarray, correct: np.ndarray) -> np.ndarray:
    rows = np.arange(len(choice))
    return correct[rows, choice]


def run_routability(
    train_cache: str | Path,
    confirmation_arm: str | Path,
    confirmation_decision: str | Path,
    output: str | Path,
) -> dict:
    cache_path = Path(train_cache).expanduser().resolve()
    arm_path = Path(confirmation_arm).expanduser().resolve()
    decision_path = Path(confirmation_decision).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    arm = _json(arm_path, "DVF1 confirmation arm")
    decision = _json(decision_path, "DVF1 confirmation decision")
    if (
        arm.get("format") != ARM_FORMAT
        or arm.get("status") != "COMPLETE"
        or arm.get("test_images_accessed") is not False
    ):
        raise RuntimeError("Confirmation arm tidak valid")
    if (
        decision.get("format") != DECISION_FORMAT
        or decision.get("decision") != "PASS"
        or decision.get("next") != "PROMOTE_DVF1_DUAL_VIEW_FUSION"
        or decision.get("training_executed") is not False
        or decision.get("test_images_accessed") is not False
    ):
        raise RuntimeError("DVF1 confirmation decision tidak valid")
    seed = str(arm["seed"])
    if decision["artifacts"][seed]["sha256"] != _sha256(arm_path):
        raise RuntimeError("Confirmation arm berbeda dari aggregate decision")
    if arm["run_contract"]["train_cache_sha256"] != _sha256(cache_path):
        raise RuntimeError("Train cache berbeda dari confirmation contract")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("split") != "train":
        raise RuntimeError("Audit hanya menerima train cache")
    if "val" in cache_path.name.lower() or "test" in cache_path.name.lower():
        raise RuntimeError("Nama cache mengindikasikan validation/test")
    arrays = _correctness(cache)
    correct = arrays["correct"]
    paper_correct = arrays["paper_correct"]
    oracle_correct = correct.any(axis=1)
    exactly_one = correct.sum(axis=1) == 1
    group_ids = [str(value) for value in cache["group_ids"]]
    folds = np.asarray([_stable_fold(value) for value in group_ids], dtype=np.int64)
    oof_choice = np.full(len(group_ids), -1, dtype=np.int64)
    fold_rows = {}
    for fold in range(FOLDS):
        train = (folds != fold) & exactly_one
        held_out = folds == fold
        targets = np.where(correct[train, 0], 1.0, -1.0)
        model = _fit_ridge_difference(arrays["difference"][train], targets)
        choice = _selector_view(
            arrays["difference"][held_out], arrays["paper_view"][held_out], model
        )
        oof_choice[held_out] = choice
        selector_correct = _evaluate_choice(choice, correct[held_out])
        baseline_correct = paper_correct[held_out]
        fold_rows[str(fold)] = {
            "pairs": int(held_out.sum()),
            "train_exactly_one_pairs": int(train.sum()),
            "paper_correct": int(baseline_correct.sum()),
            "selector_correct": int(selector_correct.sum()),
            "rescues": int(((~baseline_correct) & selector_correct).sum()),
            "regressions": int((baseline_correct & (~selector_correct)).sum()),
            "net": int(selector_correct.sum() - baseline_correct.sum()),
        }
    if bool((oof_choice < 0).any()):
        raise RuntimeError("OOF selector tidak lengkap")
    selector_correct = _evaluate_choice(oof_choice, correct)
    margin_correct = _evaluate_choice(arrays["margin_view"], correct)
    entropy_correct = _evaluate_choice(arrays["entropy_view"], correct)
    final_targets = np.where(correct[exactly_one, 0], 1.0, -1.0)
    final_model = _fit_ridge_difference(
        arrays["difference"][exactly_one], final_targets
    )
    total = len(group_ids)
    paper_count = int(paper_correct.sum())
    oracle_count = int(oracle_correct.sum())
    selector_count = int(selector_correct.sum())
    oracle_gain = oracle_count - paper_count
    selector_gain = selector_count - paper_count
    transitions = {
        "rescues": int(((~paper_correct) & selector_correct).sum()),
        "regressions": int((paper_correct & (~selector_correct)).sum()),
        "both_correct": int((paper_correct & selector_correct).sum()),
        "both_wrong": int(((~paper_correct) & (~selector_correct)).sum()),
    }
    classwise = {}
    names = {int(key): str(value) for key, value in cache["names"].items()}
    labels = arrays["labels"]
    for class_id, class_name in names.items():
        selected = labels == class_id
        classwise[class_name] = {
            "pairs": int(selected.sum()),
            "exactly_one_correct": int((selected & exactly_one).sum()),
            "paper_correct": int(paper_correct[selected].sum()),
            "selector_correct": int(selector_correct[selected].sum()),
            "net": int(selector_correct[selected].sum() - paper_correct[selected].sum()),
        }
    metrics = {
        "pairs": total,
        "exactly_one_correct_pairs": int(exactly_one.sum()),
        "neither_correct_pairs": int((correct.sum(axis=1) == 0).sum()),
        "both_correct_pairs": int((correct.sum(axis=1) == 2).sum()),
        "paper_accuracy": paper_count / total,
        "max_margin_accuracy": float(margin_correct.mean()),
        "min_entropy_accuracy": float(entropy_correct.mean()),
        "oof_selector_accuracy": selector_count / total,
        "pair_oracle_accuracy": oracle_count / total,
        "oracle_gain_over_paper": oracle_gain / total,
        "selector_gain_over_paper": selector_gain / total,
        "captured_oracle_gain": selector_gain / max(oracle_gain, 1),
    }
    criteria = {
        "pair_oracle_gain_over_paper_at_least_1_point": metrics["oracle_gain_over_paper"] >= 0.01,
        "exactly_one_correct_pairs_at_least_50": metrics["exactly_one_correct_pairs"] >= 50,
        "oof_selector_gain_over_paper_at_least_0_5_point": metrics["selector_gain_over_paper"] >= 0.005,
        "oof_selector_captures_at_least_25_percent_oracle_gain": metrics["captured_oracle_gain"] >= 0.25,
        "oof_selector_net_rescues_at_least_3": transitions["rescues"] - transitions["regressions"] >= 3,
        "positive_net_in_at_least_3_of_5_folds": sum(row["net"] > 0 for row in fold_rows.values()) >= 3,
        "validation_not_accessed": True,
        "test_not_accessed": True,
    }
    passed = all(criteria.values())
    payload = {
        "format": FORMAT,
        "scope": "grouped_train_only_oof_side_reliability",
        "settings": {
            "folds": FOLDS,
            "ridge_alpha": RIDGE_ALPHA,
            "feature_count_per_view": int(arrays["difference"].shape[1]),
            "selector_is_swap_equivariant": True,
        },
        "metrics": metrics,
        "transitions": transitions,
        "folds": fold_rows,
        "classwise": classwise,
        "final_train_selector": {
            "scale": final_model["scale"].tolist(),
            "weights": final_model["weights"].tolist(),
        },
        "criteria": criteria,
        "decision": "AUTHORIZE_DVSR1_VALIDATION_SCREEN" if passed else "STOP_SIDE_RELIABILITY_ROUTING",
        "next": "FREEZE_DVSR1_ENDPOINT_PROTOCOL" if passed else "RETAIN_CONFIRMED_DVF1",
        "contract": {
            "train_cache_sha256": _sha256(cache_path),
            "confirmation_arm_sha256": _sha256(arm_path),
            "confirmation_decision_sha256": _sha256(decision_path),
        },
        "claim_boundary": "train-only grouped OOF routability; no validation or test claim",
        "router_fitting_executed": True,
        "detector_training_executed": False,
        "validation_images_accessed": False,
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only dual-side reliability audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--confirmation-arm", required=True)
    parser.add_argument("--confirmation-decision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_routability(
        args.train_cache,
        args.confirmation_arm,
        args.confirmation_decision,
        args.output,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
