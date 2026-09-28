"""Frozen train-fitted expert router evaluated once on Defectos validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from coffee_detector.analysis.af2_box_score_factorial import (
    _match_predictions,
    summarize_detection_stats,
)
from coffee_detector.analysis.coffee_fg_diagnostics import _split_samples, _unwrap_head
from coffee_detector.analysis.defectoscafeverde_error_complementarity import (
    MODEL_ORDER,
    REFERENCE,
    validate_result_contracts,
)
from coffee_detector.analysis.defectoscafeverde_train_routability import (
    _appearance_features,
    _associate_to_anchors,
    _blank_expert,
    _kept_final,
    _native_rgb_letterbox_sample,
    fit_final_router,
    predict_final_router,
)


METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Bukan objek JSON: {source}")
    return payload


def _load_records(path: str | Path) -> list[dict]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_train_audit(summary_path: str | Path, records_path: str | Path) -> tuple[dict, list[dict], tuple[str, ...]]:
    summary = _load_json(summary_path)
    if summary.get("format") != "coffee_detector.defectoscafeverde.train_routability.v1":
        raise RuntimeError("Format audit train routability salah")
    if summary.get("settings", {}).get("input_color_order") != "RGB_native_ultralytics":
        raise RuntimeError("Audit train bukan RGB-native")
    if summary.get("decision") != "AUTHORIZE_SEPARATE_ROUTED_ARCHITECTURE_SCREEN":
        raise RuntimeError(f"Audit train tidak mengotorisasi screen: {summary.get('decision')}")
    if summary.get("validation_images_accessed") is not False or summary.get("test_images_accessed") is not False:
        raise RuntimeError("Lock validation/test audit train gagal")
    subset = tuple(summary.get("minimal_subset", {}).get("subset", []))
    if not subset or subset[0] != REFERENCE or any(model not in MODEL_ORDER for model in subset):
        raise RuntimeError(f"Subset train tidak valid: {subset}")
    records = _load_records(records_path)
    if len(records) != int(summary.get("targets", -1)):
        raise RuntimeError("Jumlah record train tidak cocok summary")
    return summary, records, subset


def _prediction_record(
    image: torch.Tensor,
    anchor_index: int,
    predictions: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    associated: Mapping[str, Mapping[int, dict]],
    subset: tuple[str, ...],
) -> dict:
    d0_boxes, d0_labels, d0_confidences = predictions[REFERENCE]
    record = {
        "anchor_available": True,
        "appearance": _appearance_features(image, d0_boxes[anchor_index]),
        "models": {
            REFERENCE: {
                "associated": True,
                "predicted_class_id": int(d0_labels[anchor_index]),
                "confidence": float(d0_confidences[anchor_index]),
                "anchor_iou": 1.0,
            }
        },
    }
    for model in subset[1:]:
        row = associated[model].get(anchor_index)
        record["models"][model] = _blank_expert() if row is None else {
            "associated": True,
            "predicted_class_id": row["predicted_class_id"],
            "confidence": row["confidence"],
            "anchor_iou": row["anchor_iou"],
        }
    return record


def route_predictions(
    image: torch.Tensor,
    predictions: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    router: Mapping[str, Any],
    association_iou: float = 0.30,
) -> tuple[torch.Tensor, Counter]:
    subset = tuple(router["subset"])
    d0_boxes, d0_labels, d0_confidences = predictions[REFERENCE]
    associated = {}
    for model in subset[1:]:
        boxes, labels, confidences = predictions[model]
        associated[model] = _associate_to_anchors(
            boxes, labels, confidences, d0_boxes, association_iou
        )
    rows = []
    choices = Counter()
    for anchor_index in range(len(d0_boxes)):
        record = _prediction_record(image, anchor_index, predictions, associated, subset)
        chosen = predict_final_router(router, record)
        selected = associated.get(chosen, {}).get(anchor_index) if chosen != REFERENCE else None
        if chosen != REFERENCE and selected is None:
            chosen = REFERENCE
        choices[chosen] += 1
        if chosen == REFERENCE:
            rows.append(torch.cat((d0_boxes[anchor_index], d0_confidences[anchor_index, None], d0_labels[anchor_index, None].float())))
        else:
            selected = associated[chosen][anchor_index]
            rows.append(torch.cat((selected["box"], d0_confidences.new_tensor([selected["confidence"], float(selected["predicted_class_id"])]))))
    if not rows:
        return d0_boxes.new_zeros((0, 6)), choices
    routed = torch.stack(rows)
    return routed[routed[:, 4].argsort(descending=True)], choices


def _stats_from_observations(observations: list[dict], names: Mapping[int, str]) -> dict:
    results = {}
    for arm in (REFERENCE, "ROUTED"):
        results[arm] = summarize_detection_stats(
            np.concatenate([np.asarray(row[arm]["tp"], dtype=bool) for row in observations], axis=0),
            np.concatenate([np.asarray(row[arm]["confidence"], dtype=float) for row in observations]),
            np.concatenate([np.asarray(row[arm]["predicted_class"], dtype=float) for row in observations]),
            np.concatenate([np.asarray(row[arm]["target_class"], dtype=float) for row in observations]),
            names,
        )
    return results


def decide_routed_screen(results: Mapping[str, Mapping[str, float]], historical_d0: Mapping[str, float]) -> dict:
    calibration = {metric: float(results[REFERENCE][metric]) - float(historical_d0[metric]) for metric in METRICS}
    deltas = {metric: float(results["ROUTED"][metric]) - float(results[REFERENCE][metric]) for metric in METRICS}
    calibration_ok = all(abs(value) <= 0.001 for value in calibration.values())
    overall = deltas["macro_map50_95"] >= 0.005 and deltas["bottom3_class_map50_95"] >= 0 and deltas["worst_class_map50_95"] >= -0.005
    tail = deltas["macro_map50_95"] >= -0.001 and deltas["bottom3_class_map50_95"] >= 0.005 and deltas["worst_class_map50_95"] >= 0
    passed = calibration_ok and (overall or tail)
    return {
        "calibration_vs_historical_d0": calibration,
        "routed_minus_runtime_d0": deltas,
        "criteria": {
            "d0_endpoint_calibrated_within_0_1_point": calibration_ok,
            "overall_route": overall,
            "lower_tail_route": tail,
            "all_12_validation_classes_present": not results["ROUTED"]["classes_without_ground_truth"],
            "detector_training_not_executed": True,
            "router_validation_fitting_not_executed": True,
            "test_not_accessed": True,
        },
        "decision": "RETAIN_ROUTED_CANDIDATE" if passed else ("INVALID_ENDPOINT_CALIBRATION" if not calibration_ok else "STOP_ROUTED_FUSION"),
        "next": "FREEZE_CONFIRMATION_AND_EFFICIENCY_PROTOCOL" if passed else "RETAIN_D0DIRECT",
    }


def run_routed_validation_screen(
    data_root: str | Path,
    result_paths: Mapping[str, str | Path],
    checkpoint_paths: Mapping[str, str | Path],
    train_summary: str | Path,
    train_records: str | Path,
    output: str | Path,
    *,
    observations_output: str | Path | None = None,
    router_output: str | Path | None = None,
    device: str = "cpu",
    image_size: int = 640,
    confidence: float = 0.001,
    association_iou: float = 0.30,
    max_det: int = 500,
) -> dict:
    endpoint_contract = validate_result_contracts(result_paths, checkpoint_paths)
    audit, train_rows, subset = validate_train_audit(train_summary, train_records)
    if audit["contract"]["dataset_audit_sha256"] != endpoint_contract["dataset_audit_sha256"]:
        raise RuntimeError("Dataset audit train dan endpoint berbeda")
    router = fit_final_router(train_rows, subset, class_count=12, alpha=10.0)
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    router_path = Path(router_output).expanduser().resolve() if router_output else destination.with_name("defectoscafeverde_ridge_router.json")
    router_path.write_text(json.dumps(router, indent=2) + "\n", encoding="utf-8")

    layout, samples = _split_samples(data_root, "val")
    if "test" in layout.splits or (Path(data_root) / "test").exists() or (Path(data_root) / "train").exists():
        raise RuntimeError("Screen runtime hanya boleh memuat validation")
    names = {int(index): str(name) for index, name in layout.names.items()}
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA tidak tersedia: {torch_device}")

    observation_path = Path(observations_output).expanduser().resolve() if observations_output else destination.with_name("defectoscafeverde_routed_validation_observations.jsonl")
    observation_path.parent.mkdir(parents=True, exist_ok=True)
    resume_contract = {
        "format": "coffee_detector.defectoscafeverde.routed_validation.resume.v1",
        "dataset_audit_sha256": endpoint_contract["dataset_audit_sha256"],
        "checkpoint_sha256": {model: endpoint_contract["models"][model]["checkpoint_sha256"] for model in subset},
        "train_summary_sha256": _sha256(train_summary),
        "train_records_sha256": _sha256(train_records),
        "router_sha256": _sha256(router_path),
        "settings": {"input_color_order": "RGB_native_ultralytics", "image_size": image_size, "confidence": confidence, "association_iou": association_iou, "max_det": max_det},
    }
    marker = observation_path.with_suffix(observation_path.suffix + ".contract.json")
    if observation_path.exists():
        if not marker.is_file() or _load_json(marker) != resume_contract:
            raise RuntimeError("Kontrak resume validation berbeda")
        observations = _load_records(observation_path)
    else:
        observation_path.write_text("", encoding="utf-8")
        marker.write_text(json.dumps(resume_contract, indent=2) + "\n", encoding="utf-8")
        observations = []
    completed = {row["image"] for row in observations}
    pending = [sample for sample in samples if str(sample[0]) not in completed]
    print(f"RESUME ROUTED VAL: {len(completed)}/{len(samples)} selesai; {len(pending)} tersisa", flush=True)

    networks = {}
    if pending:
        from ultralytics import YOLO
        for model in subset:
            network = YOLO(str(Path(checkpoint_paths[model]).resolve())).model.to(torch_device).eval()
            if int(_unwrap_head(network).nc) != len(names):
                raise RuntimeError(f"Jumlah kelas salah: {model}")
            networks[model] = network
    thresholds = torch.linspace(0.5, 0.95, 10, device=torch_device)
    choices = Counter()
    with torch.inference_mode(), observation_path.open("a", encoding="utf-8") as stream:
        for offset, (image_path, annotations) in enumerate(pending, 1):
            image, target_boxes, target_labels, _ = _native_rgb_letterbox_sample(image_path, annotations, image_size, torch_device)
            predictions = {model: _kept_final(networks[model], image, confidence, max_det) for model in subset}
            routed, image_choices = route_predictions(image, predictions, router, association_iou)
            choices.update(image_choices)
            arms = {
                REFERENCE: torch.cat((predictions[REFERENCE][0], predictions[REFERENCE][2][:, None], predictions[REFERENCE][1][:, None].float()), dim=1),
                "ROUTED": routed,
            }
            observation = {"image": str(image_path), "selection_counts": dict(image_choices)}
            for arm, prediction in arms.items():
                correct = _match_predictions(prediction[:, 5].long(), target_labels, prediction[:, :4], target_boxes, thresholds)
                observation[arm] = {
                    "tp": correct.numpy().tolist(),
                    "confidence": prediction[:, 4].float().cpu().numpy().tolist(),
                    "predicted_class": prediction[:, 5].long().cpu().numpy().tolist(),
                    "target_class": target_labels.cpu().numpy().tolist(),
                }
            stream.write(json.dumps(observation, separators=(",", ":")) + "\n"); stream.flush()
            observations.append(observation)
            done = len(completed) + offset
            if done % 50 == 0 or offset == len(pending):
                print(f"ROUTED VAL: {done}/{len(samples)}", flush=True)
    results = _stats_from_observations(observations, names)
    total_choices = Counter()
    for observation in observations:
        total_choices.update(observation.get("selection_counts", {}))
    historical_payload = _load_json(result_paths[REFERENCE])
    historical = historical_payload.get("metrics", historical_payload)
    decision = decide_routed_screen(results, historical)
    if results["ROUTED"]["classes_without_ground_truth"]:
        decision["decision"] = "INVALID_MISSING_VALIDATION_CLASS"
        decision["next"] = "RETAIN_D0DIRECT"
    payload = {
        "format": "coffee_detector.defectoscafeverde.routed_validation.v1",
        "protocol": "defectoscafeverde-routed-validation-screen-v1",
        "subset": list(subset),
        "router": {"path": str(router_path), "sha256": _sha256(router_path), "training_targets": router["training_targets"]},
        "images": len(samples),
        "results": results,
        "selection_counts": dict(total_choices),
        **decision,
        "detector_training_executed": False,
        "router_fitted_from_train_only": True,
        "test_images_accessed": False,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Defectos train-fitted routed validation screen")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-summary", required=True)
    parser.add_argument("--train-records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    for model in MODEL_ORDER:
        parser.add_argument(f"--{model.lower()}-checkpoint", required=True)
        parser.add_argument(f"--{model.lower()}-result", required=True)
    args = parser.parse_args()
    checkpoints = {model: getattr(args, f"{model.lower()}_checkpoint") for model in MODEL_ORDER}
    results = {model: getattr(args, f"{model.lower()}_result") for model in MODEL_ORDER}
    payload = run_routed_validation_screen(
        args.data_root, results, checkpoints, args.train_summary, args.train_records,
        args.output, device=args.device,
    )
    print(json.dumps({key: payload[key] for key in ("subset", "results", "calibration_vs_historical_d0", "routed_minus_runtime_d0", "criteria", "decision", "next")}, indent=2))


if __name__ == "__main__":
    main()

