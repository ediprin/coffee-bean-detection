"""Validation-only root-cause audit for RATF1's `partido` regression."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

import torch
from torchvision.ops import box_iou

from coffee_detector.analysis.coffee_fg_diagnostics import (
    _letterbox_sample,
    _rank_candidates,
    _raw_branches,
    _split_samples,
    _unwrap_head,
)
from coffee_detector.analysis.defectoscafeverde_error_complementarity import (
    _stage_outcomes,
)
from coffee_detector.data.prepare_defectoscafeverde_grouped import NAMES
from coffee_detector.defectos_ratf import RATFDetectHead


MODEL_ORDER = ("D0DIRECT", "RATF1_ACTIVE", "RATF1_ZERO_RESIDUAL")
TARGET_CLASS = "partido"
TARGET_CLASS_ID = NAMES.index(TARGET_CLASS)


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
        raise TypeError(f"{label} bukan objek JSON")
    return payload


def validate_contracts(
    d0_result: str | Path,
    ratf_result: str | Path,
    d0_checkpoint: str | Path,
    ratf_checkpoint: str | Path,
) -> dict:
    d0 = _load_json(d0_result, "D0 result")
    ratf = _load_json(ratf_result, "RATF1 result")
    if d0.get("arm") != "D0DIRECT" or ratf.get("arm") != "RATF1":
        raise RuntimeError("Arm result tidak cocok")
    for row, name in ((d0, "D0DIRECT"), (ratf, "RATF1")):
        if int(row.get("seed", -1)) != 42 or int(row.get("completed_epochs", -1)) != 50:
            raise RuntimeError(f"{name} bukan endpoint seed-42 50 epoch")
        if row.get("evaluation_split") != "val" or row.get("test_images_accessed") is not False:
            raise RuntimeError(f"Validation/test lock gagal: {name}")
    d0_contract, ratf_contract = d0["run_contract"], ratf["run_contract"]
    paired = {
        "dataset_audit_sha256_equal": d0_contract["dataset_audit_sha256"]
        == ratf_contract["dataset_audit_sha256"],
        "pretrained_checkpoint_sha256_equal": d0_contract["pretrained_checkpoint_sha256"]
        == ratf_contract["pretrained_checkpoint_sha256"],
        "native_config_sha256_equal": d0_contract["native_config_sha256"]
        == ratf_contract["native_config_sha256"],
        "training_schedule_equal": d0_contract["train"] == ratf_contract["train"],
        "same_seed": d0.get("seed") == ratf.get("seed") == 42,
        "test_not_opened": True,
    }
    if not all(paired.values()):
        raise RuntimeError(f"Paired contract gagal: {[k for k, v in paired.items() if not v]}")
    checkpoints = {
        "D0DIRECT": Path(d0_checkpoint).expanduser().resolve(),
        "RATF1": Path(ratf_checkpoint).expanduser().resolve(),
    }
    for name, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint {name}: {path}")
    if _sha256(checkpoints["D0DIRECT"]) != d0.get("checkpoint_sha256"):
        raise RuntimeError("SHA checkpoint D0 tidak cocok")
    if _sha256(checkpoints["RATF1"]) != ratf.get("checkpoint_sha256"):
        raise RuntimeError("SHA checkpoint RATF1 tidak cocok")
    return {
        "paired": paired,
        "d0_checkpoint_sha256": d0["checkpoint_sha256"],
        "ratf_checkpoint_sha256": ratf["checkpoint_sha256"],
        "test_images_accessed": False,
    }


def _localized_score_view(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    target_box: torch.Tensor,
    target_class: int,
    iou_threshold: float,
) -> dict[str, Any]:
    if not len(boxes):
        return {
            "raw_best_iou": 0.0,
            "localized_true_score": None,
            "localized_best_rival_score": None,
            "localized_margin": None,
            "localized_true_class_rank": None,
            "localized_best_rival_class_id": None,
        }
    ious = box_iou(boxes, target_box.view(1, 4)).flatten()
    best_iou = float(ious.max())
    localized = ious >= iou_threshold
    if not bool(localized.any()):
        return {
            "raw_best_iou": best_iou,
            "localized_true_score": None,
            "localized_best_rival_score": None,
            "localized_margin": None,
            "localized_true_class_rank": None,
            "localized_best_rival_class_id": None,
        }
    class_maxima = scores[localized].max(dim=0).values
    true_score = class_maxima[target_class]
    rival = class_maxima.clone()
    rival[target_class] = -torch.inf
    rival_score, rival_id = rival.max(dim=0)
    true_rank = int((class_maxima > true_score).sum()) + 1
    return {
        "raw_best_iou": best_iou,
        "localized_true_score": float(true_score),
        "localized_best_rival_score": float(rival_score),
        "localized_margin": float(true_score - rival_score),
        "localized_true_class_rank": true_rank,
        "localized_best_rival_class_id": int(rival_id),
    }


def _blank_records(samples: list[tuple[Path, tuple]]) -> list[dict]:
    records = []
    for image_path, annotations in samples:
        for target_index, annotation in enumerate(annotations):
            if int(annotation.class_id) == TARGET_CLASS_ID:
                records.append(
                    {
                        "image": str(image_path),
                        "target_index": target_index,
                        "class_id": TARGET_CLASS_ID,
                        "class_name": TARGET_CLASS,
                        "models": {},
                    }
                )
    return records


def _evaluate_network(
    network: torch.nn.Module,
    samples: list[tuple[Path, tuple]],
    records: list[dict],
    *,
    model_name: str,
    image_size: int,
    raw_count: int,
    final_confidence: float,
    iou_threshold: float,
    max_det: int,
    device: torch.device,
) -> None:
    network = network.to(device).eval()
    if int(_unwrap_head(network).nc) != len(NAMES):
        raise RuntimeError(f"Jumlah kelas salah: {model_name}")
    by_key = {(row["image"], row["target_index"]): row for row in records}
    with torch.inference_mode():
        for image_index, (image_path, annotations) in enumerate(samples, 1):
            target_indices = [
                index
                for index, annotation in enumerate(annotations)
                if int(annotation.class_id) == TARGET_CLASS_ID
            ]
            if not target_indices:
                continue
            image, target_boxes, target_labels, _ = _letterbox_sample(
                image_path, annotations, image_size, device
            )
            final, raw, head = _raw_branches(network, image, max_det)
            raw_boxes = head._get_decode_boxes(raw["one2one"]).transpose(1, 2)[0]
            raw_scores = raw["one2one"]["scores"].sigmoid().transpose(1, 2)[0]
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
            for target_index in target_indices:
                score_view = _localized_score_view(
                    raw_boxes,
                    raw_scores,
                    target_boxes[target_index],
                    TARGET_CLASS_ID,
                    iou_threshold,
                )
                by_key[(str(image_path), target_index)]["models"][model_name] = {
                    "raw_top500": raw_outcomes[target_index],
                    "final_conf0001": final_outcomes[target_index],
                    **score_view,
                }
            if image_index % 100 == 0 or image_index == len(samples):
                print(f"{model_name}: {image_index}/{len(samples)} validation images", flush=True)


def _stage_summary(records: list[dict], model: str, stage: str) -> dict:
    rows = [record["models"][model][stage] for record in records]
    matched = sum(bool(row["matched"]) for row in rows)
    correct = sum(bool(row["correct"]) for row in rows)
    wrong = [row for row in rows if row["matched"] and not row["correct"]]
    destinations = Counter(
        NAMES[int(row["predicted_class_id"])]
        for row in wrong
        if row["predicted_class_id"] is not None
    )
    return {
        "targets": len(rows),
        "accessible": sum(bool(row["accessible"]) for row in rows),
        "matched": matched,
        "correct_class": correct,
        "wrong_class": len(wrong),
        "proposal_accessibility": sum(bool(row["accessible"]) for row in rows)
        / max(len(rows), 1),
        "matched_recall": matched / max(len(rows), 1),
        "correct_decision_recall": correct / max(len(rows), 1),
        "localization_conditioned_class_accuracy": correct / max(matched, 1),
        "wrong_destinations": dict(destinations.most_common()),
    }


def _score_summary(records: list[dict], model: str) -> dict:
    rows = [record["models"][model] for record in records]
    localized = [row for row in rows if row["localized_margin"] is not None]
    margins = [float(row["localized_margin"]) for row in localized]
    ranks = [int(row["localized_true_class_rank"]) for row in localized]
    rivals = Counter(
        NAMES[int(row["localized_best_rival_class_id"])] for row in localized
    )
    return {
        "localized_targets": len(localized),
        "mean_localized_margin": mean(margins) if margins else None,
        "median_localized_margin": median(margins) if margins else None,
        "localized_top1_rate": sum(rank == 1 for rank in ranks) / max(len(ranks), 1),
        "mean_true_class_rank": mean(ranks) if ranks else None,
        "best_rival_classes": dict(rivals.most_common()),
    }


def summarize(records: list[dict]) -> dict:
    if not records:
        raise RuntimeError("Validation tidak memiliki objek partido")
    for record in records:
        if tuple(record["models"]) != MODEL_ORDER:
            raise RuntimeError("Endpoint audit tidak lengkap")
    stages = {
        stage: {
            model: _stage_summary(records, model, stage) for model in MODEL_ORDER
        }
        for stage in ("raw_top500", "final_conf0001")
    }
    scores = {model: _score_summary(records, model) for model in MODEL_ORDER}
    active_zero_iou_equal = all(
        abs(
            float(record["models"]["RATF1_ACTIVE"]["raw_best_iou"])
            - float(record["models"]["RATF1_ZERO_RESIDUAL"]["raw_best_iou"])
        )
        <= 1e-7
        for record in records
    )
    d0 = stages["raw_top500"]["D0DIRECT"]
    active = stages["raw_top500"]["RATF1_ACTIVE"]
    zero = stages["raw_top500"]["RATF1_ZERO_RESIDUAL"]
    if active["proposal_accessibility"] < d0["proposal_accessibility"] - 0.01:
        attribution = "LOCALIZATION_OR_SHARED_FEATURE_TRAINING_DEGRADATION"
    elif zero["correct_decision_recall"] > active["correct_decision_recall"]:
        attribution = "INFERENCE_TEXTURE_RESIDUAL_CONTRIBUTES_TO_CLASS_REGRESSION"
    elif zero["correct_decision_recall"] < d0["correct_decision_recall"]:
        attribution = "TRAINING_PATH_DOMINANT_ZERO_RESIDUAL_STILL_BELOW_D0"
    else:
        attribution = "AP_RANKING_EFFECT_WITHOUT_OBJECT_TOP1_REGRESSION"
    return {
        "stages": stages,
        "localized_score_diagnostic": scores,
        "raw_localization_active_zero_identical": active_zero_iou_equal,
        "attribution": attribution,
    }


def run_partido_root_cause_audit(
    data_root: str | Path,
    d0_result: str | Path,
    ratf_result: str | Path,
    d0_checkpoint: str | Path,
    ratf_checkpoint: str | Path,
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
    contract = validate_contracts(
        d0_result, ratf_result, d0_checkpoint, ratf_checkpoint
    )
    layout, samples = _split_samples(data_root, "val")
    if "test" in layout.splits or (Path(data_root) / "test").exists():
        raise RuntimeError("Test tidak boleh tersedia pada audit ini")
    if list(layout.names.values()) != NAMES:
        raise RuntimeError("Ontologi DefectosCafeVerde berubah")
    records = _blank_records(samples)
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")
    from ultralytics import YOLO

    d0_network = YOLO(str(Path(d0_checkpoint).resolve())).model
    _evaluate_network(
        d0_network, samples, records, model_name="D0DIRECT", image_size=image_size,
        raw_count=raw_count, final_confidence=final_confidence,
        iou_threshold=iou_threshold, max_det=max_det, device=torch_device,
    )
    del d0_network
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    ratf_network = YOLO(str(Path(ratf_checkpoint).resolve())).model
    if not isinstance(ratf_network.model[-1], RATFDetectHead):
        raise TypeError("Checkpoint RATF1 tidak memiliki RATFDetectHead")
    _evaluate_network(
        ratf_network, samples, records, model_name="RATF1_ACTIVE", image_size=image_size,
        raw_count=raw_count, final_confidence=final_confidence,
        iou_threshold=iou_threshold, max_det=max_det, device=torch_device,
    )
    with torch.no_grad():
        for adapter in ratf_network.model[-1].adapters:
            adapter.class_residual.weight.zero_()
            adapter.class_residual.bias.zero_()
    _evaluate_network(
        ratf_network, samples, records, model_name="RATF1_ZERO_RESIDUAL",
        image_size=image_size, raw_count=raw_count,
        final_confidence=final_confidence, iou_threshold=iou_threshold,
        max_det=max_det, device=torch_device,
    )
    summary = summarize(records)
    payload = {
        "format": "coffee_detector.defectoscafeverde.ratf1_partido_root_cause.v1",
        "scope": "validation_only_target_class_causal_inference_intervention",
        "target_class": TARGET_CLASS,
        "target_class_id": TARGET_CLASS_ID,
        "contract": contract,
        "settings": {
            "image_size": image_size,
            "raw_count": raw_count,
            "final_confidence": final_confidence,
            "iou_threshold": iou_threshold,
            "max_det": max_det,
        },
        "images": len(samples),
        "target_instances": len(records),
        **summary,
        "training_executed": False,
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--d0-result", required=True)
    parser.add_argument("--ratf-result", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--ratf-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--records-output")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    payload = run_partido_root_cause_audit(
        args.data_root,
        args.d0_result,
        args.ratf_result,
        args.d0_checkpoint,
        args.ratf_checkpoint,
        args.output,
        records_output=args.records_output,
        device=args.device,
    )
    print(json.dumps({
        "target_class": payload["target_class"],
        "target_instances": payload["target_instances"],
        "stages": payload["stages"],
        "localized_score_diagnostic": payload["localized_score_diagnostic"],
        "raw_localization_active_zero_identical": payload[
            "raw_localization_active_zero_identical"
        ],
        "attribution": payload["attribution"],
        "training_executed": False,
        "test_images_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
