"""Object-level error-complementarity audit for DefectosCafeVerde.

This module does not train, tune, ensemble, or access test.  It evaluates
completed seed-42 validation checkpoints on exactly the same ground-truth
objects and asks a deliberately narrower question: do the candidate models
ever correct objects that the native detector gets wrong?
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torchvision.ops import box_iou

from coffee_detector.analysis.coffee_fg_diagnostics import (
    _confidence_ordered_match,
    _greedy_match,
    _letterbox_sample,
    _rank_candidates,
    _raw_branches,
    _split_samples,
    _unwrap_head,
)


MODEL_ORDER = ("D0DIRECT", "AF2DIRECT", "DCWCF1", "LIFRPF1", "RAFC1")
REFERENCE = "D0DIRECT"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label}: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} bukan objek JSON: {source}")
    return payload


def validate_result_contracts(
    result_paths: Mapping[str, str | Path],
    checkpoint_paths: Mapping[str, str | Path],
) -> dict:
    """Validate that all five endpoints belong to one development contract."""

    if tuple(result_paths) != MODEL_ORDER or tuple(checkpoint_paths) != MODEL_ORDER:
        raise ValueError(f"Urutan model harus tepat {MODEL_ORDER}")
    rows: dict[str, dict] = {}
    dataset_shas: set[str] = set()
    for model in MODEL_ORDER:
        row = _load_json(result_paths[model], f"Result {model}")
        if row.get("arm") != model or int(row.get("seed", -1)) != 42:
            raise RuntimeError(f"Arm/seed result tidak cocok: {model}")
        if row.get("evaluation_split") != "val":
            raise RuntimeError(f"{model} bukan endpoint validation")
        if row.get("test_images_accessed") is not False:
            raise RuntimeError(f"Test lock gagal: {model}")
        if int(row.get("completed_epochs", -1)) != 50:
            raise RuntimeError(f"{model} bukan run 50 epoch lengkap")
        contract = row.get("run_contract", {})
        dataset_sha = contract.get("dataset_audit_sha256")
        if not isinstance(dataset_sha, str) or not dataset_sha:
            raise RuntimeError(f"Dataset audit SHA hilang: {model}")
        dataset_shas.add(dataset_sha)
        checkpoint = Path(checkpoint_paths[model]).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint {model}: {checkpoint}")
        observed_sha = _sha256(checkpoint)
        if observed_sha != row.get("checkpoint_sha256"):
            raise RuntimeError(f"Checkpoint SHA tidak cocok: {model}")
        rows[model] = row
    if len(dataset_shas) != 1:
        raise RuntimeError(f"Dataset audit SHA tidak seragam: {dataset_shas}")
    return {
        "dataset_audit_sha256": next(iter(dataset_shas)),
        "seed": 42,
        "epochs": 50,
        "split": "val",
        "models": {
            model: {
                "checkpoint_sha256": rows[model]["checkpoint_sha256"],
                "protocol": rows[model].get("protocol")
                or rows[model].get("run_contract", {}).get("protocol"),
            }
            for model in MODEL_ORDER
        },
        "test_images_accessed": False,
    }


def _blank_stage() -> dict[str, Any]:
    return {
        "accessible": False,
        "matched": False,
        "correct": False,
        "predicted_class_id": None,
        "confidence": None,
        "iou": None,
    }


def _stage_outcomes(
    predicted_boxes: torch.Tensor,
    predicted_labels: torch.Tensor,
    predicted_confidences: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    iou_threshold: float,
    confidence_ordered: bool,
) -> list[dict[str, Any]]:
    outcomes = [_blank_stage() for _ in range(len(target_boxes))]
    if not len(target_boxes):
        return outcomes
    matrix = (
        box_iou(predicted_boxes, target_boxes)
        if len(predicted_boxes)
        else target_boxes.new_zeros((0, len(target_boxes)))
    )
    if len(matrix):
        accessible = matrix.max(dim=0).values >= iou_threshold
        for target_index in accessible.nonzero().flatten().tolist():
            outcomes[int(target_index)]["accessible"] = True
    matches = (
        _confidence_ordered_match(
            predicted_boxes,
            predicted_confidences,
            target_boxes,
            iou_threshold,
        )
        if confidence_ordered
        else _greedy_match(predicted_boxes, target_boxes, iou_threshold)
    )
    for prediction, target, iou in matches:
        expected = int(target_labels[target])
        actual = int(predicted_labels[prediction])
        outcomes[target] = {
            "accessible": outcomes[target]["accessible"],
            "matched": True,
            "correct": actual == expected,
            "predicted_class_id": actual,
            "confidence": float(predicted_confidences[prediction]),
            "iou": float(iou),
        }
    return outcomes


def _evaluate_checkpoint(
    checkpoint: str | Path,
    samples: list[tuple[Path, tuple]],
    records: list[dict],
    *,
    model_name: str,
    class_count: int,
    image_size: int,
    raw_count: int,
    final_confidence: float,
    iou_threshold: float,
    max_det: int,
    device: torch.device,
) -> None:
    from ultralytics import YOLO

    network = YOLO(str(Path(checkpoint).expanduser().resolve())).model.to(device).eval()
    if int(_unwrap_head(network).nc) != class_count:
        raise RuntimeError(f"Jumlah kelas checkpoint salah: {model_name}")
    cursor = 0
    with torch.inference_mode():
        for image_index, (image_path, annotations) in enumerate(samples, 1):
            image, target_boxes, target_labels, _ = _letterbox_sample(
                image_path, annotations, image_size, device
            )
            final, raw, head = _raw_branches(network, image, max_det)
            raw_boxes, raw_scores = head._get_decode_boxes(raw["one2one"]).transpose(1, 2)[0], raw[
                "one2one"
            ]["scores"].sigmoid().transpose(1, 2)[0]
            selected_boxes, selected_labels, selected_confidences = _rank_candidates(
                raw_boxes, raw_scores, raw_count
            )
            raw_outcomes = _stage_outcomes(
                selected_boxes,
                selected_labels,
                selected_confidences,
                target_boxes,
                target_labels,
                iou_threshold=iou_threshold,
                confidence_ordered=False,
            )
            kept = final[final[:, 4] >= final_confidence]
            final_outcomes = _stage_outcomes(
                kept[:, :4],
                kept[:, 5].long(),
                kept[:, 4],
                target_boxes,
                target_labels,
                iou_threshold=iou_threshold,
                confidence_ordered=True,
            )
            for target_index in range(len(target_boxes)):
                record = records[cursor + target_index]
                if record["image"] != str(image_path) or record["target_index"] != target_index:
                    raise RuntimeError("Urutan target berubah antarcheckpoint")
                record["models"][model_name] = {
                    "raw_top500": raw_outcomes[target_index],
                    "final_conf0001": final_outcomes[target_index],
                }
            cursor += len(target_boxes)
            if image_index % 100 == 0 or image_index == len(samples):
                print(
                    f"{model_name}: {image_index}/{len(samples)} validation images",
                    flush=True,
                )
    del network
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _model_summary(records: list[dict], model: str, stage: str) -> dict:
    targets = len(records)
    rows = [record["models"][model][stage] for record in records]
    accessible = sum(bool(row["accessible"]) for row in rows)
    matched = sum(bool(row["matched"]) for row in rows)
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "targets": targets,
        "accessible": accessible,
        "matched": matched,
        "correct_class": correct,
        "wrong_or_missed": targets - correct,
        "proposal_accessibility": accessible / max(targets, 1),
        "matched_recall": matched / max(targets, 1),
        "correct_decision_recall": correct / max(targets, 1),
        "localization_conditioned_class_accuracy": correct / max(matched, 1),
    }


def summarize_records(
    records: list[dict],
    names: Mapping[int, str],
    *,
    reference: str = REFERENCE,
) -> dict:
    """Summarize per-target records without recomputing inference."""

    if not records:
        raise ValueError("Tidak ada target untuk diringkas")
    model_names = tuple(records[0]["models"])
    if model_names != MODEL_ORDER or reference not in model_names:
        raise ValueError("Set/urutan model pada record tidak sesuai protokol")
    stages = ("raw_top500", "final_conf0001")
    result: dict[str, Any] = {"stages": {}}
    for stage in stages:
        model_rows = {
            model: _model_summary(records, model, stage) for model in model_names
        }
        reference_correct = {
            index
            for index, record in enumerate(records)
            if record["models"][reference][stage]["correct"]
        }
        correct_sets = {
            model: {
                index
                for index, record in enumerate(records)
                if record["models"][model][stage]["correct"]
            }
            for model in model_names
        }
        union_correct = set().union(*correct_sets.values())
        candidates = {}
        for model in model_names[1:]:
            rescued = correct_sets[model] - reference_correct
            regressed = reference_correct - correct_sets[model]
            candidates[model] = {
                "rescued_reference_errors": len(rescued),
                "lost_reference_correct": len(regressed),
                "net_correct_objects": len(correct_sets[model]) - len(reference_correct),
                "correct_intersection": len(correct_sets[model] & reference_correct),
                "correct_union": len(correct_sets[model] | reference_correct),
                "correct_jaccard": len(correct_sets[model] & reference_correct)
                / max(len(correct_sets[model] | reference_correct), 1),
            }
        signatures = Counter(
            "+".join(
                model
                for model in model_names
                if record["models"][model][stage]["correct"]
            )
            or "NONE"
            for record in records
        )
        per_class = {}
        for class_id, class_name in names.items():
            indices = {
                index
                for index, record in enumerate(records)
                if int(record["class_id"]) == int(class_id)
            }
            reference_class_correct = reference_correct & indices
            union_class_correct = union_correct & indices
            per_class[class_name] = {
                "targets": len(indices),
                "correct_by_model": {
                    model: len(correct_sets[model] & indices) for model in model_names
                },
                "reference_correct_decision_recall": len(reference_class_correct)
                / max(len(indices), 1),
                "union_correct_decision_recall": len(union_class_correct)
                / max(len(indices), 1),
                "union_gain_over_reference": (
                    len(union_class_correct) - len(reference_class_correct)
                )
                / max(len(indices), 1),
                "rescued_reference_errors_by_model": {
                    model: len((correct_sets[model] - reference_correct) & indices)
                    for model in model_names[1:]
                },
            }
        result["stages"][stage] = {
            "models": model_rows,
            "candidate_vs_reference": candidates,
            "correct_set_signatures": dict(signatures.most_common()),
            "union": {
                "correct_objects": len(union_correct),
                "correct_decision_recall": len(union_correct) / len(records),
                "gain_over_reference": (
                    len(union_correct) - len(reference_correct)
                )
                / len(records),
                "remaining_errors": len(records) - len(union_correct),
            },
            "per_class": per_class,
        }
    return result


def _build_records(samples: list[tuple[Path, tuple]], names: Mapping[int, str]) -> list[dict]:
    records = []
    for image_path, annotations in samples:
        for target_index, annotation in enumerate(annotations):
            records.append(
                {
                    "image": str(image_path),
                    "target_index": target_index,
                    "class_id": int(annotation.class_id),
                    "class_name": names[int(annotation.class_id)],
                    "models": {},
                }
            )
    return records


def run_error_complementarity_audit(
    data_root: str | Path,
    result_paths: Mapping[str, str | Path],
    checkpoint_paths: Mapping[str, str | Path],
    output: str | Path,
    *,
    records_output: str | Path | None = None,
    device: str = "cpu",
    image_size: int = 640,
    raw_count: int = 500,
    final_confidence: float = 0.001,
    iou_threshold: float = 0.5,
    max_det: int = 500,
) -> dict:
    contract = validate_result_contracts(result_paths, checkpoint_paths)
    layout, samples = _split_samples(data_root, "val")
    if "test" in layout.splits or (Path(data_root) / "test").exists():
        raise RuntimeError("Test tidak boleh tersedia pada audit ini")
    names = {int(index): str(name) for index, name in layout.names.items()}
    if len(names) != 12:
        raise RuntimeError(f"Ontologi harus 12 kelas, ditemukan {len(names)}")
    records = _build_records(samples, names)
    if not records:
        raise RuntimeError("Validation tidak memiliki target")
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")
    for model in MODEL_ORDER:
        _evaluate_checkpoint(
            checkpoint_paths[model],
            samples,
            records,
            model_name=model,
            class_count=len(names),
            image_size=image_size,
            raw_count=raw_count,
            final_confidence=final_confidence,
            iou_threshold=iou_threshold,
            max_det=max_det,
            device=torch_device,
        )
    summary = summarize_records(records, names)
    d0_result = _load_json(result_paths[REFERENCE], "D0 result")
    class_ap = d0_result.get("map50_95_by_class", {})
    if set(class_ap) != set(names.values()):
        raise RuntimeError("D0 result tidak memuat AP seluruh 12 kelas")
    worst_class = min(class_ap, key=class_ap.get)
    final = summary["stages"]["final_conf0001"]
    worst = final["per_class"][worst_class]
    criteria = {
        "final_union_gain_at_least_1_point": final["union"]["gain_over_reference"]
        >= 0.01,
        "worst_class_has_any_rescue": sum(
            worst["rescued_reference_errors_by_model"].values()
        )
        > 0,
        "all_results_same_dataset_audit": True,
        "all_runs_seed42_50_epochs_validation": True,
        "training_not_executed": True,
        "test_not_accessed": True,
    }
    if criteria["final_union_gain_at_least_1_point"] and criteria[
        "worst_class_has_any_rescue"
    ]:
        decision = "COMPLEMENTARY_INFORMATION_PRESENT_ROUTER_AUDIT_ONLY"
        next_step = "FREEZE_TRAIN_ONLY_ROUTABILITY_AUDIT_BEFORE_ANY_NEW_MODEL"
    elif criteria["final_union_gain_at_least_1_point"]:
        decision = "CLASS_SELECTIVE_COMPLEMENTARITY_WITHOUT_WORST_CLASS_REPAIR"
        next_step = "DO_NOT_CLAIM_GENERAL_REPAIR; REVIEW_RESCUED_CLASSES_ONLY"
    else:
        decision = "NO_MATERIAL_COMPLEMENTARY_HEADROOM"
        next_step = "STOP_FUSION_AND_RETAIN_D0DIRECT"
    payload = {
        "format": "coffee_detector.defectoscafeverde.error_complementarity.v1",
        "scope": "validation_only_object_level_information_audit",
        "contract": contract,
        "settings": {
            "image_size": image_size,
            "raw_count": raw_count,
            "final_confidence": final_confidence,
            "iou_threshold": iou_threshold,
            "max_det": max_det,
        },
        "images": len(samples),
        "targets": len(records),
        "reference": REFERENCE,
        "reference_worst_class_by_ap": worst_class,
        **summary,
        "criteria": criteria,
        "decision": decision,
        "next": next_step,
        "claim_boundary": (
            "descriptive validation-only oracle audit; union uses ground truth and is "
            "not a deployable ensemble or superiority estimate"
        ),
        "training_executed": False,
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    record_path = (
        Path(records_output).expanduser().resolve()
        if records_output is not None
        else destination.with_name(destination.stem + "_targets.jsonl")
    )
    with record_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    payload["target_records"] = str(record_path)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DefectosCafeVerde object-level checkpoint complementarity audit"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--records-output")
    parser.add_argument("--device", default="cpu")
    for model in MODEL_ORDER:
        key = model.lower()
        parser.add_argument(f"--{key}-checkpoint", required=True)
        parser.add_argument(f"--{key}-result", required=True)
    args = parser.parse_args()
    checkpoints = {
        model: getattr(args, f"{model.lower()}_checkpoint") for model in MODEL_ORDER
    }
    results = {model: getattr(args, f"{model.lower()}_result") for model in MODEL_ORDER}
    payload = run_error_complementarity_audit(
        args.data_root,
        results,
        checkpoints,
        args.output,
        records_output=args.records_output,
        device=args.device,
    )
    print(json.dumps({
        "decision": payload["decision"],
        "next": payload["next"],
        "targets": payload["targets"],
        "worst_class": payload["reference_worst_class_by_ap"],
        "final_union": payload["stages"]["final_conf0001"]["union"],
        "criteria": payload["criteria"],
    }, indent=2))


if __name__ == "__main__":
    main()
