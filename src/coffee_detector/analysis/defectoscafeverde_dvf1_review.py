"""Validation-only pair-transition and bootstrap review for DVF1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from coffee_detector.defectos_dual_view import (
    DualViewFusionConfig,
    SymmetricDualViewFuser,
)
from coffee_detector.defectos_dual_view.model import paper_base_logits
from coffee_detector.experiments.run_defectoscafeverde_dual_view_fusion import (
    FORMAT as RESULT_FORMAT,
    _metrics,
)


FORMAT = "coffee_detector.defectoscafeverde.dvf1.review.v1"
ITERATIONS = 10_000
BOOTSTRAP_SEED = 20_261_006
METRICS = (
    "physical_pair_accuracy",
    "macro_class_accuracy",
    "bottom3_class_accuracy",
    "worst_class_accuracy",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label}: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} bukan objek JSON")
    return payload


def _mcnemar_exact(rescues: int, regressions: int) -> float:
    discordant = rescues + regressions
    if not discordant:
        return 1.0
    lower = min(rescues, regressions)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _correctness_metrics(
    correct: np.ndarray, labels: np.ndarray, class_ids: list[int]
) -> dict[str, float]:
    per_class = []
    for class_id in class_ids:
        selected = labels == class_id
        per_class.append(float(correct[selected].mean()))
    ordered = sorted(per_class)
    return {
        "physical_pair_accuracy": float(correct.mean()),
        "macro_class_accuracy": float(np.mean(ordered)),
        "bottom3_class_accuracy": float(np.mean(ordered[:3])),
        "worst_class_accuracy": float(ordered[0]),
    }


def stratified_pair_bootstrap(
    paper_correct: np.ndarray,
    candidate_correct: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int = ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    class_ids = sorted(int(value) for value in np.unique(labels))
    groups = {class_id: np.flatnonzero(labels == class_id) for class_id in class_ids}
    point_paper = _correctness_metrics(paper_correct, labels, class_ids)
    point_candidate = _correctness_metrics(candidate_correct, labels, class_ids)
    distributions = {metric: np.empty(iterations, dtype=np.float64) for metric in METRICS}
    rng = np.random.default_rng(seed)
    for iteration in range(iterations):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups.values()]
        )
        sampled_labels = labels[indices]
        paper = _correctness_metrics(paper_correct[indices], sampled_labels, class_ids)
        candidate = _correctness_metrics(
            candidate_correct[indices], sampled_labels, class_ids
        )
        for metric in METRICS:
            distributions[metric][iteration] = candidate[metric] - paper[metric]
    result = {}
    for metric in METRICS:
        values = distributions[metric]
        result[metric] = {
            "point_delta": point_candidate[metric] - point_paper[metric],
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "probability_positive": float((values > 0).mean()),
            "probability_nonnegative": float((values >= 0).mean()),
        }
    return {
        "iterations": iterations,
        "seed": seed,
        "stratified_by_class": True,
        "metrics": result,
    }


def _load_artifacts(
    result_path: Path, checkpoint_path: Path, cache_path: Path
) -> tuple[dict, dict, dict, SymmetricDualViewFuser]:
    result = _json(result_path, "DVF1 result")
    if (
        result.get("format") != RESULT_FORMAT
        or result.get("decision") != "PASS"
        or result.get("detector_training_executed") is not False
        or result.get("test_images_accessed") is not False
    ):
        raise RuntimeError("DVF1 result bukan PASS yang valid")
    if not checkpoint_path.is_file() or not cache_path.is_file():
        raise FileNotFoundError("Checkpoint atau validation cache DVF1 tidak ditemukan")
    if _sha256(checkpoint_path) != result["artifacts"]["fuser_checkpoint_sha256"]:
        raise RuntimeError("SHA checkpoint fuser tidak cocok")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 100:
        raise RuntimeError("Checkpoint fuser bukan epoch 100")
    if cache.get("split") != "val" or cache.get("contract", {}).get(
        "test_images_accessed"
    ) is not False:
        raise RuntimeError("Validation cache tidak memenuhi kontrak")
    for key, value in result["run_contract"].items():
        if key in checkpoint.get("contract", {}) and checkpoint["contract"][key] != value:
            raise RuntimeError(f"Kontrak checkpoint berbeda: {key}")
    config = DualViewFusionConfig.from_mapping(checkpoint["config"])
    fuser = SymmetricDualViewFuser(config)
    fuser.load_state_dict(checkpoint["model"])
    return result, checkpoint, cache, fuser.eval()


def _pair_outputs(cache: dict, fuser: SymmetricDualViewFuser) -> dict:
    logits = cache["view_logits"].float()
    ious = cache["ious"].float()
    labels = cache["labels"].long()
    scores = logits.sigmoid()
    rows = torch.arange(len(labels))
    view_confidence, view_class = scores.max(dim=2)
    paper_view = view_confidence.argmax(dim=1)
    paper_class = view_class[rows, paper_view]
    paper_correct = (paper_class == labels) & (ious[rows, paper_view] >= 0.5)
    with torch.inference_mode():
        base = paper_base_logits(logits)
        fused = fuser(logits)
    fused_class = fused.argmax(dim=1)
    class_support = scores.gather(
        2, fused_class[:, None, None].expand(-1, 2, 1)
    ).squeeze(2)
    fused_view = class_support.argmax(dim=1)
    fused_correct = (fused_class == labels) & (ious[rows, fused_view] >= 0.5)
    fused_top2 = fused.softmax(dim=1).topk(2, dim=1).values
    return {
        "labels": labels,
        "paper_class": paper_class,
        "paper_view": paper_view,
        "paper_correct": paper_correct,
        "fused_class": fused_class,
        "fused_view": fused_view,
        "fused_correct": fused_correct,
        "fused_margin": fused_top2[:, 0] - fused_top2[:, 1],
        "residual_max_abs": (fused - base).abs().amax(dim=1),
    }


def run_review(
    result_path: str | Path,
    checkpoint_path: str | Path,
    cache_path: str | Path,
    output: str | Path,
    *,
    records_output: str | Path | None = None,
    iterations: int = ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    result_path = Path(result_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    cache_path = Path(cache_path).expanduser().resolve()
    result, checkpoint, cache, fuser = _load_artifacts(
        result_path, checkpoint_path, cache_path
    )
    outputs = _pair_outputs(cache, fuser)
    labels = outputs["labels"]
    paper_correct = outputs["paper_correct"]
    fused_correct = outputs["fused_correct"]
    names = {int(key): str(value) for key, value in cache["names"].items()}
    paper_metrics = _metrics(paper_correct, labels, names)
    fused_metrics = _metrics(fused_correct, labels, names)
    calibration = {
        "paper": {
            metric: paper_metrics[metric]
            - result["values"]["PAPER_MAX_CONFIDENCE"][metric]
            for metric in METRICS
        },
        "DVF1": {
            metric: fused_metrics[metric] - result["values"]["DVF1"][metric]
            for metric in METRICS
        },
    }
    transition_names = []
    for paper, candidate in zip(paper_correct.tolist(), fused_correct.tolist()):
        transition_names.append(
            "both_correct"
            if paper and candidate
            else "rescue"
            if (not paper and candidate)
            else "regression"
            if (paper and not candidate)
            else "both_wrong"
        )
    transitions = Counter(transition_names)
    classwise = {}
    net_by_class = {}
    records = []
    for index, transition in enumerate(transition_names):
        class_id = int(labels[index])
        class_name = names[class_id]
        records.append(
            {
                "physical_group_id": cache["group_ids"][index],
                "class_id": class_id,
                "class_name": class_name,
                "transition": transition,
                "paper_predicted_class": names[int(outputs["paper_class"][index])],
                "dvf1_predicted_class": names[int(outputs["fused_class"][index])],
                "paper_view": int(outputs["paper_view"][index]),
                "dvf1_view": int(outputs["fused_view"][index]),
                "dvf1_margin": float(outputs["fused_margin"][index]),
                "residual_max_abs": float(outputs["residual_max_abs"][index]),
            }
        )
    for class_id, class_name in names.items():
        selected = [row for row in records if row["class_id"] == class_id]
        counts = Counter(row["transition"] for row in selected)
        net = counts["rescue"] - counts["regression"]
        net_by_class[class_name] = net
        classwise[class_name] = {
            "groups": len(selected),
            "both_correct": counts["both_correct"],
            "rescues": counts["rescue"],
            "regressions": counts["regression"],
            "both_wrong": counts["both_wrong"],
            "net_correct_pairs": net,
        }
    bootstrap = stratified_pair_bootstrap(
        paper_correct.numpy(),
        fused_correct.numpy(),
        labels.numpy(),
        iterations=iterations,
        seed=seed,
    )
    criteria = {
        "saved_endpoints_reproduced_exactly": all(
            value == 0.0
            for endpoint in calibration.values()
            for value in endpoint.values()
        ),
        "net_rescues_at_least_3_pairs": transitions["rescue"]
        - transitions["regression"]
        >= 3,
        "regressions_no_more_than_2_pairs": transitions["regression"] <= 2,
        "no_class_loses_more_than_1_net_pair": min(net_by_class.values()) >= -1,
        "bootstrap_probability_positive_pair_gain_at_least_0_80": bootstrap[
            "metrics"
        ]["physical_pair_accuracy"]["probability_positive"]
        >= 0.80,
        "detector_training_not_executed": result["detector_training_executed"]
        is False,
        "review_training_not_executed": True,
        "test_not_opened": result["test_images_accessed"] is False,
    }
    passed = all(criteria.values())
    payload = {
        "format": FORMAT,
        "scope": "validation_only_pair_transition_review",
        "transitions": dict(transitions),
        "net_correct_pairs": transitions["rescue"] - transitions["regression"],
        "mcnemar_exact_two_sided_p": _mcnemar_exact(
            transitions["rescue"], transitions["regression"]
        ),
        "classwise": classwise,
        "bootstrap": bootstrap,
        "endpoint_calibration": calibration,
        "criteria": criteria,
        "decision": "AUTHORIZE_DVF1_FUSER_MULTISEED"
        if passed
        else "STOP_DVF1_AT_SEED42",
        "next": "FREEZE_SEEDS_123_AND_2026"
        if passed
        else "REPORT_SEED42_AS_EXPLORATORY_ONLY",
        "claim_boundary": (
            "reused-validation diagnostic; no model training and no locked-test inference"
        ),
        "artifacts": {
            "result_sha256": _sha256(result_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "validation_cache_sha256": _sha256(cache_path),
            "checkpoint_epoch": checkpoint["epoch"],
        },
        "training_executed": False,
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if records_output is not None:
        record_path = Path(records_output).expanduser().resolve()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="DVF1 validation-only pair review")
    parser.add_argument("--result", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--records-output")
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    result = run_review(
        args.result,
        args.checkpoint,
        args.validation_cache,
        args.output,
        records_output=args.records_output,
        iterations=args.iterations,
        seed=args.seed,
    )
    print("TRANSITIONS:", result["transitions"])
    print("NET CORRECT:", result["net_correct_pairs"])
    print("MCNEMAR P:", result["mcnemar_exact_two_sided_p"])
    print("CLASSWISE:", result["classwise"])
    print("BOOTSTRAP:", result["bootstrap"])
    print("CALIBRATION:", result["endpoint_calibration"])
    print("CRITERIA:", result["criteria"])
    print("DECISION:", result["decision"])
    print("NEXT:", result["next"])
    print("TRAINING:", result["training_executed"], "| TEST:", result["test_images_accessed"])


if __name__ == "__main__":
    main()
