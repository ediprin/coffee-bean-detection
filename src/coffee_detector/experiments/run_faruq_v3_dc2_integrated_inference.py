"""Detection-level DC2 integrated inference screen on locked validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.dc2_crop.evaluation import detection_map_summary
from coffee_detector.dc2_crop.integrated import (
    classify_integrated_predictions,
    collect_all_detector_predictions,
    collect_ground_truth_records,
    decide_dc2_integrated,
    extract_integrated_global_descriptors,
)
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def run_dc2_integrated_inference_screening(
    data_root: str | Path,
    grouped_summary: str | Path,
    detector_checkpoint: str | Path,
    dc2c_summary: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str = "0",
    batch_size: int = 64,
    workers: int = 2,
    authorize_evaluation: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("DC2d integrated screening dikunci untuk seed 42")
    if not authorize_evaluation:
        raise RuntimeError("Gunakan --authorize-evaluation setelah protocol/CI dibekukan")

    data_root = Path(data_root).expanduser().resolve()
    detector_checkpoint = Path(detector_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Locked holdout tidak boleh tersedia pada DC2d")
    audit = audit_dataset(data_root, output_root / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit development dataset gagal")

    dc2c = _load_json(dc2c_summary, "DC2c summary")
    if dc2c.get("decision") != "PASS":
        raise RuntimeError("DC2c MSFA belum PASS; integrated escalation tidak diotorisasi")
    if dc2c.get("test_images_accessed") is not False:
        raise RuntimeError("DC2c summary tidak memenuhi test-lock contract")
    resolution = int(dc2c["resolution_from_dc2b"])
    classifier_checkpoint = Path(
        dc2c["results"]["MSFA"]["checkpoint"]
    ).expanduser().resolve()
    if not classifier_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint MSFA tidak ditemukan: {classifier_checkpoint}")

    ground_truth, names, _images = collect_ground_truth_records(data_root, "val")
    if len(names) != 21:
        raise RuntimeError(f"DC2d mengharapkan 21 kelas, diterima {len(names)}")
    predictions, prediction_names, prediction_meta = collect_all_detector_predictions(
        detector_checkpoint,
        data_root,
        "val",
        output_root / "cache/all_val_predictions.json",
        device=device,
    )
    if prediction_names != names:
        raise RuntimeError("Ontology detector dan GT tidak konsisten")
    if not predictions:
        raise RuntimeError("Detector tidak menghasilkan prediction pada val")

    global_descriptors, global_meta = extract_integrated_global_descriptors(
        detector_checkpoint,
        predictions,
        output_root / "cache/all_val_global.npz",
        split="val",
        device=device,
    )
    refined, classifier_meta = classify_integrated_predictions(
        predictions,
        global_descriptors,
        classifier_checkpoint,
        resolution,
        device=device,
        batch_size=batch_size,
        workers=workers,
    )
    if len(refined) != len(predictions):
        raise RuntimeError("Refined prediction count berubah; box-set harus identik")
    box_identity = all(
        left.image_path == right.image_path
        and left.predicted_xyxy == right.predicted_xyxy
        and left.predicted_confidence == right.predicted_confidence
        for left, right in zip(predictions, refined)
    )
    if not box_identity:
        raise RuntimeError("DC2d mengubah geometry/confidence detector")

    native_metrics = detection_map_summary(predictions, ground_truth, len(names))
    refined_metrics = detection_map_summary(refined, ground_truth, len(names))
    decision = decide_dc2_integrated(native_metrics, refined_metrics)
    payload = {
        "protocol": "faruq-v3-dc2-integrated-inference-screening-v1",
        "stage": "broad_search_detection_level_integrated_inference",
        "seed": seed,
        "evaluation_split": "val_only",
        "test_images_accessed": False,
        "test_opened": False,
        "training_executed": False,
        "paper_basis": {
            "testing": "DC2 Section III-C: crop detected regions and refine detector category predictions with the fine-grained classifier.",
            "global_local": "DC2 Eqs. (6)-(8): local raw-instance features plus cropped global-stream features.",
            "not_implemented_here": "DC2 Eq. (9) joint end-to-end optimization and IFE Eqs. (2)-(3).",
        },
        "adaptation_boundary": (
            "Frozen YOLO26 detector; class-agnostic NMS approximates the single parent-category detector. "
            "All predicted boxes, including unmatched false positives, are re-read as raw RGB crops. "
            "P3/P4/P5 global descriptors use the previously screened YOLO26 MSFA adaptation. "
            "Detector box geometry and confidence are preserved; only fine-grained class ID is replaced. "
            "This is integrated inference screening, not literal end-to-end DC2 reproduction."
        ),
        "resolution": resolution,
        "detector_checkpoint": str(detector_checkpoint),
        "classifier_checkpoint": str(classifier_checkpoint),
        "prediction_cache": prediction_meta,
        "global_cache": global_meta,
        "classifier": classifier_meta,
        "static_gates": {
            "same_prediction_count": len(refined) == len(predictions),
            "box_and_detector_confidence_preserved": box_identity,
            "all_21_classes_present_in_gt": len({item.class_id for item in ground_truth}) == 21,
            "test_locked": not (data_root / "test").exists(),
        },
        "results": {
            "NATIVE": native_metrics,
            "DC2_INTEGRATED": refined_metrics,
        },
        "class_names": {str(key): value for key, value in names.items()},
        **decision,
    }
    summary = output_root / "dc2_integrated_inference_screening.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 DC2 integrated inference screen")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--dc2c-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--authorize-evaluation", action="store_true")
    args = parser.parse_args()
    result = run_dc2_integrated_inference_screening(
        args.data_root,
        args.grouped_summary,
        args.detector_checkpoint,
        args.dc2c_summary,
        args.output_root,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        authorize_evaluation=args.authorize_evaluation,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
