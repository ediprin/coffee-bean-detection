from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torchvision.ops import box_iou

from coffee_detector.af2_complement.modules import SpaceFrequencySelectionResidual, low_high_split
from coffee_detector.analysis.coffee_fg_diagnostics import (
    _confidence_ordered_match,
    _decode_branch,
    _letterbox_sample,
    _rank_candidates,
    _raw_branches,
    _split_samples,
    _unwrap_head,
)


VARIANTS = ("AF2CTRL", "AF2SFS1", "AF2SFS1_BYPASS", "AF2SFS1_SPATIAL", "AF2SFS1_FREQUENCY")
SELECTOR_FIELDS = (
    "spatial_weight",
    "frequency_weight",
    "selector_entropy",
    "spatial_energy",
    "frequency_energy",
    "frequency_contribution_fraction",
    "residual_input_ratio",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: str | Path, label: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_arm_report(payload: dict, arm: str) -> None:
    if payload.get("format") != "coffee_detector.af2_complement.arm_result.v1":
        raise RuntimeError(f"Report {arm} memakai schema yang salah")
    if payload.get("arm") != arm or int(payload.get("seed", -1)) != 42:
        raise RuntimeError(f"Kontrak arm/seed salah untuk {arm}")
    if payload.get("test_images_accessed") is not False:
        raise RuntimeError(f"Report {arm} tidak membuktikan test terkunci")


def _train_size_thresholds(data_root: Path) -> tuple[float, float]:
    _layout, samples = _split_samples(data_root, "train")
    areas = np.asarray(
        [item.width * item.height for _path, annotations in samples for item in annotations],
        dtype=np.float64,
    )
    if len(areas) < 3 or not np.isfinite(areas).all():
        raise RuntimeError("Area train tidak cukup untuk mendefinisikan size bins")
    low, high = np.quantile(areas, [1.0 / 3.0, 2.0 / 3.0])
    if not 0 < low < high:
        raise RuntimeError("Train-derived size thresholds tidak valid")
    return float(low), float(high)


def _size_name(area: float, thresholds: tuple[float, float]) -> str:
    if area < thresholds[0]:
        return "small"
    if area < thresholds[1]:
        return "medium"
    return "large"


def _target_records(
    network: torch.nn.Module,
    image: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    raw_count: int,
    confidence: float,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    final, raw, head = _raw_branches(network, image, max_det=500)
    raw_boxes, raw_scores = _decode_branch(head, raw["one2one"])
    ranked_boxes, _ranked_labels, _ranked_confidences = _rank_candidates(
        raw_boxes, raw_scores, raw_count
    )
    if len(ranked_boxes) and len(target_boxes):
        raw_iou = box_iou(ranked_boxes, target_boxes).max(dim=0).values
    else:
        raw_iou = target_boxes.new_zeros(len(target_boxes))

    kept = final[final[:, 4] >= confidence]
    matches = _confidence_ordered_match(
        kept[:, :4], kept[:, 4], target_boxes, iou_threshold
    )
    by_target = {target: (prediction, iou) for prediction, target, iou in matches}
    rows = []
    for target_index in range(len(target_boxes)):
        match = by_target.get(target_index)
        expected = int(target_labels[target_index])
        predicted = int(kept[match[0], 5]) if match is not None else None
        correct = bool(predicted == expected) if predicted is not None else False
        rows.append(
            {
                "raw_accessible": bool(float(raw_iou[target_index]) >= iou_threshold),
                "raw_max_iou": float(raw_iou[target_index]),
                "final_matched": match is not None,
                "final_iou": float(match[1]) if match is not None else 0.0,
                "predicted_class": predicted,
                "correct_class": correct,
                "outcome": "correct" if correct else ("wrong" if match is not None else "miss"),
            }
        )
    return rows


@contextmanager
def _adapter_intervention(adapter: SpaceFrequencySelectionResidual, mode: str) -> Iterator[None]:
    handles = []
    if mode == "bypass":
        handles.append(adapter.register_forward_hook(lambda _m, args, _out: args[0]))
    elif mode in {"spatial", "frequency"}:
        selected_index = 0 if mode == "spatial" else 1

        def force_selector(_module, _args, output):
            forced = torch.full_like(output, -20.0)
            forced[:, selected_index] = 20.0
            return forced

        handles.append(adapter.selector.register_forward_hook(force_selector))
    elif mode != "normal":
        raise ValueError(f"Intervensi tidak dikenal: {mode}")
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _capture_adapter(adapter: SpaceFrequencySelectionResidual) -> Iterator[dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module, args):
        captured["input"] = args[0].detach()

    def post_hook(_module, _args, output):
        captured["output"] = output.detach()

    handles = [adapter.register_forward_pre_hook(pre_hook), adapter.register_forward_hook(post_hook)]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _region(tensor: torch.Tensor, box: torch.Tensor, image_size: int) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    x1 = max(0, min(width - 1, int(math.floor(float(box[0]) * width / image_size))))
    y1 = max(0, min(height - 1, int(math.floor(float(box[1]) * height / image_size))))
    x2 = max(x1 + 1, min(width, int(math.ceil(float(box[2]) * width / image_size))))
    y2 = max(y1 + 1, min(height, int(math.ceil(float(box[3]) * height / image_size))))
    return tensor[..., y1:y2, x1:x2]


def _selector_records(
    adapter: SpaceFrequencySelectionResidual,
    captured: dict[str, torch.Tensor],
    target_boxes: torch.Tensor,
    *,
    image_size: int,
) -> list[dict[str, float]]:
    value = captured["input"].detach()
    output = captured["output"].detach()
    _low, high = low_high_split(value, adapter.kernel_size)
    spatial = adapter.spatial(value).detach()
    weights = torch.softmax(adapter.selector(value), dim=1).detach()
    eps = torch.finfo(value.dtype).eps
    rows = []
    for box in target_boxes:
        value_roi = _region(value, box, image_size)
        output_roi = _region(output, box, image_size)
        spatial_roi = _region(spatial, box, image_size)
        high_roi = _region(high, box, image_size)
        weights_roi = _region(weights, box, image_size)
        spatial_weight = float(weights_roi[:, 0].mean())
        frequency_weight = float(weights_roi[:, 1].mean())
        entropy = float(
            (-(weights_roi.clamp_min(eps) * weights_roi.clamp_min(eps).log()).sum(dim=1)).mean()
        )
        spatial_contribution = float((weights_roi[:, :1] * spatial_roi).abs().mean())
        frequency_contribution = float((weights_roi[:, 1:] * high_roi).abs().mean())
        contribution_total = spatial_contribution + frequency_contribution
        input_rms = float(value_roi.square().mean().sqrt())
        residual_rms = float((output_roi - value_roi).square().mean().sqrt())
        rows.append(
            {
                "spatial_weight": spatial_weight,
                "frequency_weight": frequency_weight,
                "selector_entropy": entropy,
                "spatial_energy": float(spatial_roi.square().mean().sqrt()),
                "frequency_energy": float(high_roi.square().mean().sqrt()),
                "frequency_contribution_fraction": float(
                    frequency_contribution / max(contribution_total, eps)
                ),
                "residual_input_ratio": float(residual_rms / max(input_rms, eps)),
            }
        )
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    targets = len(rows)
    matched = sum(bool(row["final_matched"]) for row in rows)
    correct = sum(bool(row["correct_class"]) for row in rows)
    return {
        "targets": targets,
        "raw_proposal_accessibility": float(sum(bool(row["raw_accessible"]) for row in rows) / max(targets, 1)),
        "mean_raw_max_iou": float(np.mean([row["raw_max_iou"] for row in rows])) if rows else 0.0,
        "final_matched_recall": float(matched / max(targets, 1)),
        "mean_final_matched_iou": float(np.mean([row["final_iou"] for row in rows if row["final_matched"]])) if matched else 0.0,
        "conditional_top1_accuracy": float(correct / max(matched, 1)),
        "correct_decision_recall": float(correct / max(targets, 1)),
    }


def _delta(candidate: dict, control: dict) -> dict[str, float]:
    return {
        key: float(candidate[key] - control[key])
        for key in (
            "raw_proposal_accessibility",
            "mean_raw_max_iou",
            "final_matched_recall",
            "mean_final_matched_iou",
            "conditional_top1_accuracy",
            "correct_decision_recall",
        )
    }


def _selector_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"objects": len(rows)}
    for field in SELECTOR_FIELDS:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        result[field] = {
            "mean": float(values.mean()) if len(values) else 0.0,
            "median": float(np.median(values)) if len(values) else 0.0,
            "q25": float(np.quantile(values, 0.25)) if len(values) else 0.0,
            "q75": float(np.quantile(values, 0.75)) if len(values) else 0.0,
            "std": float(values.std()) if len(values) else 0.0,
        }
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    x, y = _rankdata(np.asarray(left, dtype=np.float64)), _rankdata(np.asarray(right, dtype=np.float64))
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run_af2sfs1_root_cause(
    control_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    control_report: str | Path,
    candidate_report: str | Path,
    data_root: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 640,
    raw_count: int = 500,
    confidence: float = 0.25,
    iou_threshold: float = 0.50,
) -> dict:
    from ultralytics import YOLO

    control_checkpoint = Path(control_checkpoint).expanduser().resolve()
    candidate_checkpoint = Path(candidate_checkpoint).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    if not control_checkpoint.is_file() or not candidate_checkpoint.is_file():
        raise FileNotFoundError("Checkpoint control/candidate tidak lengkap")
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    control_payload = _read(control_report, "AF2CTRL report")
    candidate_payload = _read(candidate_report, "AF2SFS1 report")
    _validate_arm_report(control_payload, "AF2CTRL")
    _validate_arm_report(candidate_payload, "AF2SFS1")
    for checkpoint, report, arm in (
        (control_checkpoint, control_payload, "AF2CTRL"),
        (candidate_checkpoint, candidate_payload, "AF2SFS1"),
    ):
        expected_run = f"{arm}_seed42"
        if checkpoint.parent.name != "weights" or checkpoint.parent.parent.name != expected_run:
            raise RuntimeError(f"Checkpoint input bukan milik run {expected_run}")
        recorded = report.get("checkpoint")
        if recorded:
            recorded_path = Path(recorded)
            if recorded_path.parent.name != "weights" or recorded_path.parent.parent.name != expected_run:
                raise RuntimeError(f"Checkpoint report bukan milik run {expected_run}")

    layout, samples = _split_samples(data_root, "val")
    if len(samples) != 294 or len(layout.names) != 21:
        raise RuntimeError("Kontrak Faruq-v3 validation harus 294 gambar dan 21 kelas")
    thresholds = _train_size_thresholds(data_root)
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")
    control = YOLO(str(control_checkpoint)).model.to(torch_device).eval()
    candidate = YOLO(str(candidate_checkpoint)).model.to(torch_device).eval()
    expected_names = {int(index): str(name) for index, name in layout.names.items()}
    for network, arm in ((control, "AF2CTRL"), (candidate, "AF2SFS1")):
        names = {int(index): str(name) for index, name in network.names.items()}
        if names != expected_names or int(_unwrap_head(network).nc) != 21:
            raise RuntimeError(f"Ontology checkpoint {arm} tidak cocok")
    adapter = candidate.model[-1].adapter
    if not isinstance(adapter, SpaceFrequencySelectionResidual):
        raise TypeError("Checkpoint candidate tidak mengekspos SpaceFrequencySelectionResidual")

    rows: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    selector_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for image_index, (image_path, annotations) in enumerate(samples, 1):
            image, target_boxes, target_labels, _shape = _letterbox_sample(
                image_path, annotations, image_size, torch_device
            )
            base = [
                {
                    "image": str(image_path.relative_to(layout.root)),
                    "target_index": index,
                    "class_id": int(target_labels[index]),
                    "class_name": expected_names[int(target_labels[index])],
                    "normalized_area": float(item.width * item.height),
                    "size_bin": _size_name(float(item.width * item.height), thresholds),
                }
                for index, item in enumerate(annotations)
            ]

            control_records = _target_records(
                control, image, target_boxes, target_labels,
                raw_count=raw_count, confidence=confidence, iou_threshold=iou_threshold,
            )
            rows["AF2CTRL"].extend([{**meta, **record} for meta, record in zip(base, control_records)])

            with _capture_adapter(adapter) as captured:
                normal_records = _target_records(
                    candidate, image, target_boxes, target_labels,
                    raw_count=raw_count, confidence=confidence, iou_threshold=iou_threshold,
                )
            rows["AF2SFS1"].extend([{**meta, **record} for meta, record in zip(base, normal_records)])
            observed = _selector_records(adapter, captured, target_boxes, image_size=image_size)
            selector_rows.extend([{**meta, **values} for meta, values in zip(base, observed)])

            for variant, intervention in (
                ("AF2SFS1_BYPASS", "bypass"),
                ("AF2SFS1_SPATIAL", "spatial"),
                ("AF2SFS1_FREQUENCY", "frequency"),
            ):
                with _adapter_intervention(adapter, intervention):
                    records = _target_records(
                        candidate, image, target_boxes, target_labels,
                        raw_count=raw_count, confidence=confidence, iou_threshold=iou_threshold,
                    )
                rows[variant].extend([{**meta, **record} for meta, record in zip(base, records)])
            if image_index % 25 == 0 or image_index == len(samples):
                print(f"ROOT-CAUSE AUDIT {image_index}/{len(samples)}", flush=True)

    for index, (control_row, candidate_row, selector_row) in enumerate(
        zip(rows["AF2CTRL"], rows["AF2SFS1"], selector_rows)
    ):
        transition = f'{control_row["outcome"]}_to_{candidate_row["outcome"]}'
        rows["AF2CTRL"][index]["paired_transition"] = transition
        rows["AF2SFS1"][index]["paired_transition"] = transition
        selector_row["paired_transition"] = transition

    global_metrics = {variant: _summary(values) for variant, values in rows.items()}
    intervention_deltas = {
        variant: _delta(global_metrics[variant], global_metrics["AF2SFS1_BYPASS"])
        for variant in ("AF2SFS1", "AF2SFS1_SPATIAL", "AF2SFS1_FREQUENCY")
    }
    grouped: dict[str, Any] = {}
    for group_field, group_values in (
        ("class_name", list(expected_names.values())),
        ("size_bin", ["small", "medium", "large"]),
    ):
        grouped[group_field] = {}
        for group in group_values:
            control_group = [row for row in rows["AF2CTRL"] if row[group_field] == group]
            candidate_group = [row for row in rows["AF2SFS1"] if row[group_field] == group]
            control_summary, candidate_summary = _summary(control_group), _summary(candidate_group)
            grouped[group_field][group] = {
                "AF2CTRL": control_summary,
                "AF2SFS1": candidate_summary,
                "delta": _delta(candidate_summary, control_summary),
            }

    control_ap = control_payload["metrics"]["map50_95_by_class"]
    candidate_ap = candidate_payload["metrics"]["map50_95_by_class"]
    per_class_ap = [
        {
            "class_name": name,
            "AF2CTRL": float(control_ap[name]),
            "AF2SFS1": float(candidate_ap[name]),
            "delta": float(candidate_ap[name] - control_ap[name]),
        }
        for name in expected_names.values()
    ]
    per_class_ap.sort(key=lambda row: row["delta"], reverse=True)

    selector_by_class = {
        name: _selector_summary([row for row in selector_rows if row["class_name"] == name])
        for name in expected_names.values()
    }
    selector_by_size = {
        size: _selector_summary([row for row in selector_rows if row["size_bin"] == size])
        for size in ("small", "medium", "large")
    }
    transitions = Counter(row["paired_transition"] for row in selector_rows)
    selector_by_transition = {
        transition: _selector_summary(
            [row for row in selector_rows if row["paired_transition"] == transition]
        )
        for transition in sorted(transitions)
    }

    class_lookup = {row["class_name"]: row for row in per_class_ap}
    correlations = {}
    for diagnostic_name in (
        "raw_proposal_accessibility",
        "final_matched_recall",
        "conditional_top1_accuracy",
        "correct_decision_recall",
    ):
        correlations[f"ap_delta_vs_{diagnostic_name}_delta"] = _spearman(
            [class_lookup[name]["delta"] for name in expected_names.values()],
            [grouped["class_name"][name]["delta"][diagnostic_name] for name in expected_names.values()],
        )
    for selector_name in ("frequency_weight", "frequency_contribution_fraction", "residual_input_ratio"):
        correlations[f"ap_delta_vs_{selector_name}"] = _spearman(
            [class_lookup[name]["delta"] for name in expected_names.values()],
            [selector_by_class[name][selector_name]["mean"] for name in expected_names.values()],
        )

    main_delta = _delta(global_metrics["AF2SFS1"], global_metrics["AF2CTRL"])
    signals = {
        "raw_localization": main_delta["raw_proposal_accessibility"],
        "final_selection": main_delta["final_matched_recall"],
        "conditional_classification": main_delta["conditional_top1_accuracy"],
    }
    positive = {name: value for name, value in signals.items() if value > 0}
    attribution = max(positive, key=positive.get).upper() if positive else "NO_POSITIVE_DIAGNOSTIC_SIGNAL"
    selector_normal_gain = intervention_deltas["AF2SFS1"]["correct_decision_recall"]
    active_selector_supported = selector_normal_gain > 0

    selector_weight_error = max(
        abs(row["spatial_weight"] + row["frequency_weight"] - 1.0)
        for row in selector_rows
    )
    gates = {
        "arm_and_seed_contract_exact": True,
        "same_21_class_ontology": True,
        "validation_images_exactly_294": len(samples) == 294,
        "all_variants_same_target_count": len({len(value) for value in rows.values()}) == 1,
        "selector_weights_finite": all(
            math.isfinite(row[field]) for row in selector_rows for field in SELECTOR_FIELDS
        ),
        "selector_weights_sum_to_one": selector_weight_error <= 1e-5,
        "training_not_executed": True,
        "test_not_accessed": True,
    }
    payload = {
        "format": "coffee_detector.af2sfs1.root_cause.v1",
        "protocol": "faruq-v3-af2sfs1-root-cause-2026-08-28",
        "checkpoints": {
            "AF2CTRL": {"path": str(control_checkpoint), "sha256": _sha256(control_checkpoint)},
            "AF2SFS1": {"path": str(candidate_checkpoint), "sha256": _sha256(candidate_checkpoint)},
        },
        "configuration": {
            "image_size": image_size,
            "raw_candidate_count": raw_count,
            "confidence_threshold": confidence,
            "iou_threshold": iou_threshold,
            "size_thresholds_from_train_normalized_area": {
                "q33": thresholds[0], "q67": thresholds[1]
            },
        },
        "headline_ap": {
            "AF2CTRL": {
                key: float(control_payload["metrics"][key])
                for key in ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
            },
            "AF2SFS1": {
                key: float(candidate_payload["metrics"][key])
                for key in ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
            },
        },
        "per_class_ap": per_class_ap,
        "top5_improved_classes": per_class_ap[:5],
        "top5_regressed_classes": list(reversed(per_class_ap[-5:])),
        "paired_diagnostic": {
            "global": global_metrics,
            "AF2SFS1_minus_AF2CTRL": main_delta,
            "by_class": grouped["class_name"],
            "by_size": grouped["size_bin"],
            "outcome_transitions": dict(sorted(transitions.items())),
        },
        "inference_interventions": {
            "reference": "AF2SFS1_BYPASS",
            "deltas_vs_bypass": intervention_deltas,
            "interpretation": {
                "active_selector_supported": active_selector_supported,
                "normal_correct_decision_recall_gain_vs_bypass": selector_normal_gain,
            },
        },
        "selector_observability": {
            "global": _selector_summary(selector_rows),
            "by_class": selector_by_class,
            "by_size": selector_by_size,
            "by_transition": selector_by_transition,
            "maximum_weight_sum_error": selector_weight_error,
        },
        "correlations": correlations,
        "root_cause_attribution": {
            "dominant_observed_signal": attribution,
            "signals": signals,
            "active_selector_inference_effect_supported": active_selector_supported,
            "boundary": "post-hoc paired validation attribution; not a replacement model-selection result",
        },
        "gates": gates,
        "decision": "INTERPRETABLE" if all(gates.values()) else "INVALID",
        "training_executed": False,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["summary"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="AF2SFS1 validation-only root-cause diagnostic")
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--control-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_af2sfs1_root_cause(
        args.control_checkpoint,
        args.candidate_checkpoint,
        args.control_report,
        args.candidate_report,
        args.data_root,
        args.output,
        device=args.device,
    )
    print(json.dumps(result["root_cause_attribution"], indent=2, ensure_ascii=False))
    print("GATES:", result["gates"])
    print("DECISION:", result["decision"])
    print("SUMMARY:", result["summary"])


if __name__ == "__main__":
    main()
