"""Validation-only D0FT/AF2 box-score factorial.

The study swaps the raw one-to-one regression and classification tensors at
the same fixed YOLO26 anchor/grid indices.  It does not average weights, train
parameters, use decoded boxes as a second-stage input, or access test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torchvision.ops import box_iou

from coffee_detector.analysis.coffee_fg_diagnostics import (
    _unwrap_head,
)
from coffee_detector.dataset import discover_layout
from coffee_detector.experiments.run_faruq_v3_baseline import (
    load_faruq_grouped_summary,
)


PROTOCOL = "faruq-v3-af2-box-score-factorial-validation-v1"
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)
ARMS = {
    "DD": {"box_source": "D0FT", "score_source": "D0FT"},
    "DA": {"box_source": "D0FT", "score_source": "AF2"},
    "AD": {"box_source": "AF2", "score_source": "D0FT"},
    "AA": {"box_source": "AF2", "score_source": "AF2"},
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combine_branch(
    box_branch: Mapping[str, torch.Tensor],
    score_branch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Combine aligned raw branch tensors without mutating either source."""

    required = {"boxes", "scores", "feats"}
    if not required <= set(box_branch) or not required <= set(score_branch):
        raise KeyError("Branch harus memuat boxes, scores, dan feats")
    boxes, scores = box_branch["boxes"], score_branch["scores"]
    if boxes.ndim != 3 or scores.ndim != 3 or boxes.shape[0] != scores.shape[0]:
        raise ValueError("Shape batch branch tidak kompatibel")
    if boxes.shape[-1] != scores.shape[-1]:
        raise ValueError("Jumlah anchor/grid boxes dan scores berbeda")
    box_feature_shapes = [tuple(value.shape) for value in box_branch["feats"]]
    score_feature_shapes = [tuple(value.shape) for value in score_branch["feats"]]
    if box_feature_shapes != score_feature_shapes:
        raise ValueError("Pyramid feature layout boxes dan scores berbeda")
    # Anchor generation and DFL decode belong to the regression source.  The
    # score source contributes logits only; its feature values are not reused.
    return {"boxes": boxes, "scores": scores, "feats": box_branch["feats"]}


def _postprocess_branch(head, branch: Mapping[str, torch.Tensor], max_det: int) -> torch.Tensor:
    head.max_det = int(max_det)
    decoded = head._inference(dict(branch)).permute(0, 2, 1)
    return head.postprocess(decoded)


def _network_raw(model: torch.nn.Module, images: torch.Tensor, max_det: int):
    head = model.model[-1]
    base_head = _unwrap_head(model)
    head.max_det = int(max_det)
    base_head.max_det = int(max_det)
    output = model(images)
    if not isinstance(output, tuple) or not isinstance(output[1], dict):
        raise TypeError("Checkpoint tidak menyediakan raw one-to-one/one-to-many branches")
    final, raw = output
    if not {"one2one", "one2many"} <= set(raw):
        raise KeyError("Raw checkpoint tidak memuat kedua branch YOLO26")
    return final, raw, base_head


def _validation_loader(
    yolo_model,
    yaml_path: Path,
    *,
    image_size: int,
    batch_size: int,
    workers: int,
):
    """Build the same rectangular, RGB validation loader used by model.val()."""

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    overrides = dict(getattr(yolo_model, "overrides", {}))
    overrides.update(
        {
            "data": str(yaml_path),
            "imgsz": int(image_size),
            "batch": int(batch_size),
            "workers": int(workers),
            "rect": True,
            "cache": False,
            "augment": False,
            "mode": "val",
            "task": "detect",
        }
    )
    args = get_cfg(overrides=overrides)
    data = check_det_dataset(str(yaml_path))
    dataset = build_yolo_dataset(
        args,
        data["val"],
        int(batch_size),
        data,
        mode="val",
        stride=32,
    )
    return dataset, build_dataloader(
        dataset,
        batch=int(batch_size),
        workers=int(workers),
        shuffle=False,
        rank=-1,
    )


def _match_predictions(
    predicted_classes: torch.Tensor,
    target_classes: torch.Tensor,
    predicted_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """Ultralytics-compatible greedy class-aware matching at ten IoUs."""

    correct = np.zeros((len(predicted_classes), len(thresholds)), dtype=bool)
    if not len(predicted_classes) or not len(target_classes):
        return torch.from_numpy(correct)
    iou = box_iou(target_boxes, predicted_boxes)
    iou = iou * (target_classes[:, None] == predicted_classes[None, :])
    matrix = iou.detach().cpu().numpy()
    for index, threshold in enumerate(thresholds.detach().cpu().tolist()):
        matches = np.array(np.nonzero(matrix >= threshold)).T
        if not len(matches):
            continue
        if len(matches) > 1:
            matches = matches[matrix[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), index] = True
    return torch.from_numpy(correct)


def summarize_detection_stats(
    true_positives: np.ndarray,
    confidences: np.ndarray,
    predicted_classes: np.ndarray,
    target_classes: np.ndarray,
    names: Mapping[int, str],
) -> dict[str, Any]:
    from ultralytics.utils.metrics import ap_per_class

    result = ap_per_class(
        true_positives,
        confidences,
        predicted_classes,
        target_classes,
        plot=False,
        names=dict(names),
    )
    _, _, precision, recall, _, ap, class_indices, *_ = result
    class_indices = np.asarray(class_indices, dtype=int).reshape(-1)
    ap = np.asarray(ap, dtype=np.float64)
    if ap.ndim == 1:
        ap = ap[:, None]
    by_class = {
        names[int(class_id)]: float(ap[row].mean())
        for row, class_id in enumerate(class_indices)
        if int(class_id) in names
    }
    missing_ids = sorted(set(names) - set(map(int, class_indices)))
    values = np.asarray(list(by_class.values()), dtype=np.float64)
    if not len(values):
        raise RuntimeError("Evaluator tidak menghasilkan AP kelas")
    return {
        "map50_95": float(ap.mean()),
        "map50": float(ap[:, 0].mean()),
        "precision": float(np.asarray(precision, dtype=float).mean()),
        "recall": float(np.asarray(recall, dtype=float).mean()),
        "macro_map50_95": float(values.mean()),
        "bottom3_class_map50_95": float(np.sort(values)[:3].mean()),
        "worst_class_map50_95": float(values.min()),
        "worst_class": min(by_class, key=by_class.get),
        "map50_95_by_class": by_class,
        "classes_without_ground_truth": [names[index] for index in missing_ids],
    }


def _reference_seed42(path: str | Path) -> tuple[dict, dict[str, dict[str, float]]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Reference summary tidak ditemukan: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("protocol")
        != "faruq-v3-af2-igem-paired-validation-confirmation-v1"
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
        or payload.get("test_opened") is not False
        or "42" not in payload.get("per_seed", {})
    ):
        raise RuntimeError("Reference bukan konfirmasi AF2 validation-only yang kompatibel")
    row = payload["per_seed"]["42"]
    return payload, {
        model: {metric: float(row[model][metric]) for metric in METRICS}
        for model in ("D0FT", "AF2")
    }


def _model_names(network: torch.nn.Module) -> dict[int, str]:
    names = getattr(network, "names", None)
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict):
        raise RuntimeError("Checkpoint tidak menyimpan class names")
    return {int(key): str(value) for key, value in names.items()}


def _device(value: str) -> torch.device:
    device = torch.device(f"cuda:{value}" if str(value).isdigit() else value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {device}")
    return device


def _finalize_stats(stats: dict[str, list[np.ndarray]], names: Mapping[int, str]) -> dict:
    return summarize_detection_stats(
        np.concatenate(stats["tp"], axis=0),
        np.concatenate(stats["confidence"], axis=0),
        np.concatenate(stats["predicted_class"], axis=0),
        np.concatenate(stats["target_class"], axis=0),
        names,
    )


def _decision(results: Mapping[str, Mapping[str, float]]) -> tuple[dict, str, str]:
    target_delta = {
        metric: float(results["DA"][metric]) - float(results["AA"][metric])
        for metric in METRICS
    }
    reverse_delta = {
        metric: float(results["AD"][metric]) - float(results["DD"][metric])
        for metric in METRICS
    }
    criteria = {
        "d0ft_boxes_af2_scores_macro_within_0_1_point_of_af2": target_delta[
            "macro_map50_95"
        ]
        >= -0.001 - 1e-12,
        "d0ft_boxes_af2_scores_bottom3_not_lower_than_af2": target_delta[
            "bottom3_class_map50_95"
        ]
        >= 0.0,
        "d0ft_boxes_af2_scores_worst_not_lower_than_af2": target_delta[
            "worst_class_map50_95"
        ]
        >= 0.0,
    }
    if all(criteria.values()):
        decision = "SUPPORT_D0FT_BOX_AF2_SCORE_ARCHITECTURE"
        next_action = "IMPLEMENT_DECOUPLED_AF2_CLASSIFICATION_BRANCH"
    elif all(target_delta[metric] < 0.0 for metric in METRICS):
        decision = "AF2_BOX_SCORE_INTERACTION_NECESSARY"
        next_action = "RETAIN_AF2_AND_STOP_NAIVE_BRANCH_SEPARATION"
    else:
        decision = "MIXED_BOX_SCORE_INTERACTION"
        next_action = "DIAGNOSE_PER_CLASS_BEFORE_ARCHITECTURE_CHANGE"
    return {
        "DA_minus_AA": target_delta,
        "AD_minus_DD": reverse_delta,
        "criteria": criteria,
    }, decision, next_action


def run_af2_box_score_factorial(
    d0ft_checkpoint: str | Path,
    af2_checkpoint: str | Path,
    data_root: str | Path,
    grouped_summary: str | Path,
    reference_summary: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 640,
    batch_size: int = 16,
    workers: int = 2,
    max_det: int = 500,
    confidence: float = 0.001,
    endpoint_tolerance: float = 0.002,
) -> dict:
    from ultralytics import YOLO

    d0ft_checkpoint = Path(d0ft_checkpoint).expanduser().resolve()
    af2_checkpoint = Path(af2_checkpoint).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    for path in (d0ft_checkpoint, af2_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    reference_payload, references = _reference_seed42(reference_summary)
    layout = discover_layout(data_root)
    if "val" not in layout.splits:
        raise FileNotFoundError(f"Validation tidak ditemukan di {data_root}")
    torch_device = _device(device)
    yolo_models = {"D0FT": YOLO(str(d0ft_checkpoint)), "AF2": YOLO(str(af2_checkpoint))}
    networks = {
        source: model.model.to(torch_device).eval() for source, model in yolo_models.items()
    }
    dataset, loader = _validation_loader(
        yolo_models["D0FT"],
        layout.yaml_path,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
    )
    names = {source: _model_names(model) for source, model in networks.items()}
    heads = {source: _unwrap_head(model) for source, model in networks.items()}
    gates: dict[str, bool] = {
        "same_21_class_ontology": names["D0FT"] == names["AF2"] == layout.names
        and len(layout.names) == 21,
        "same_head_class_count": int(heads["D0FT"].nc) == int(heads["AF2"].nc) == 21,
        "same_stride": torch.equal(
            heads["D0FT"].stride.detach().cpu(), heads["AF2"].stride.detach().cpu()
        ),
        "validation_only": True,
        "training_not_executed": True,
        "test_not_accessed": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Static ontology/head gate gagal: {gates}")

    stats = {
        arm: {key: [] for key in ("tp", "confidence", "predicted_class", "target_class")}
        for arm in ARMS
    }
    thresholds = torch.linspace(0.5, 0.95, 10, device=torch_device)
    alignment: dict[str, Any] = {}
    endpoint_max_diff = {"DD": 0.0, "AA": 0.0}
    raw_score_difference = 0.0
    raw_box_difference = 0.0

    with torch.inference_mode():
        completed = 0
        for batch in loader:
            images = batch["img"].to(torch_device, non_blocking=True).float().div_(255.0)
            batch_indices = batch["batch_idx"].to(torch_device)
            batch_boxes = batch["bboxes"].to(torch_device)
            batch_classes = batch["cls"].to(torch_device).long().reshape(-1)
            final: dict[str, torch.Tensor] = {}
            raw: dict[str, dict] = {}
            for source, network in networks.items():
                final[source], raw[source], heads[source] = _network_raw(
                    network, images, max_det
                )
            branches = {source: raw[source]["one2one"] for source in networks}
            shape_gate = (
                branches["D0FT"]["boxes"].shape == branches["AF2"]["boxes"].shape
                and branches["D0FT"]["scores"].shape
                == branches["AF2"]["scores"].shape
                and branches["D0FT"]["boxes"].shape[-1]
                == branches["D0FT"]["scores"].shape[-1]
            )
            if not shape_gate:
                raise RuntimeError("Raw anchor/grid tensor D0FT dan AF2 tidak sejajar")
            raw_score_difference = max(
                raw_score_difference,
                float((branches["D0FT"]["scores"] - branches["AF2"]["scores"]).abs().max()),
            )
            raw_box_difference = max(
                raw_box_difference,
                float((branches["D0FT"]["boxes"] - branches["AF2"]["boxes"]).abs().max()),
            )

            arm_predictions = {}
            for arm, sources in ARMS.items():
                branch = combine_branch(
                    branches[sources["box_source"]], branches[sources["score_source"]]
                )
                arm_predictions[arm] = _postprocess_branch(
                    heads[sources["box_source"]], branch, max_det
                )
                if arm in endpoint_max_diff:
                    native_source = "D0FT" if arm == "DD" else "AF2"
                    endpoint_max_diff[arm] = max(
                        endpoint_max_diff[arm],
                        float((arm_predictions[arm] - final[native_source]).abs().max()),
                    )

            height, width = images.shape[-2:]
            scale = torch.tensor([width, height, width, height], device=torch_device)
            for sample_index in range(len(images)):
                target_mask = batch_indices == sample_index
                target_classes = batch_classes[target_mask]
                target_boxes = batch_boxes[target_mask]
                if len(target_boxes):
                    from ultralytics.utils import ops

                    target_boxes = ops.xywh2xyxy(target_boxes) * scale
                for arm in ARMS:
                    prediction = arm_predictions[arm][sample_index]
                    # Matches Ultralytics end-to-end NMS filtering exactly.
                    prediction = prediction[prediction[:, 4] > confidence]
                    predicted_boxes = prediction[:, :4]
                    predicted_classes = prediction[:, 5].long()
                    correct = _match_predictions(
                        predicted_classes,
                        target_classes,
                        predicted_boxes,
                        target_boxes,
                        thresholds,
                    )
                    stats[arm]["tp"].append(correct.numpy())
                    stats[arm]["confidence"].append(
                        prediction[:, 4].float().cpu().numpy()
                    )
                    stats[arm]["predicted_class"].append(
                        predicted_classes.cpu().numpy()
                    )
                    stats[arm]["target_class"].append(target_classes.cpu().numpy())
            completed += len(images)
            if completed % 50 < len(images) or completed == len(dataset):
                print(f"FACTORIAL {completed}/{len(dataset)}", flush=True)

    results = {arm: _finalize_stats(value, layout.names) for arm, value in stats.items()}
    calibration = {
        "DD_vs_historical_D0FT": {
            metric: float(results["DD"][metric]) - references["D0FT"][metric]
            for metric in METRICS
        },
        "AA_vs_historical_AF2": {
            metric: float(results["AA"][metric]) - references["AF2"][metric]
            for metric in METRICS
        },
    }
    gates.update(
        {
            "raw_shapes_aligned": True,
            "raw_score_swap_is_active": raw_score_difference > 0.0,
            "raw_box_swap_is_active": raw_box_difference > 0.0,
            "pure_d0ft_postprocess_exact": endpoint_max_diff["DD"] == 0.0,
            "pure_af2_postprocess_exact": endpoint_max_diff["AA"] == 0.0,
            "pure_d0ft_metric_calibrated": all(
                abs(value) <= endpoint_tolerance
                for value in calibration["DD_vs_historical_D0FT"].values()
            ),
            "pure_af2_metric_calibrated": all(
                abs(value) <= endpoint_tolerance
                for value in calibration["AA_vs_historical_AF2"].values()
            ),
            "all_21_validation_classes_present": all(
                not result["classes_without_ground_truth"] for result in results.values()
            ),
        }
    )
    audit_pass = all(gates.values())
    if audit_pass:
        comparison, decision, next_action = _decision(results)
    else:
        comparison, decision, next_action = (
            {"criteria": {}},
            "INVALID_EVALUATOR_OR_ALIGNMENT",
            "STOP_WITHOUT_ARCHITECTURE_CHANGE",
        )
    payload = {
        "format": "coffee_detector.af2_box_score_factorial.result.v1",
        "protocol": PROTOCOL,
        "seed": 42,
        "arms": ARMS,
        "results": results,
        "historical_reference": references,
        "calibration": calibration,
        "endpoint_postprocess_max_abs_diff": endpoint_max_diff,
        "raw_tensor_max_abs_difference": {
            "boxes": raw_box_difference,
            "scores": raw_score_difference,
        },
        "gates": gates,
        "comparison": comparison,
        "decision": decision,
        "next": next_action,
        "checkpoints": {
            "D0FT": str(d0ft_checkpoint),
            "AF2": str(af2_checkpoint),
        },
        "checkpoint_sha256": {
            "D0FT": _sha256(d0ft_checkpoint),
            "AF2": _sha256(af2_checkpoint),
        },
        "reference_summary": str(Path(reference_summary).expanduser().resolve()),
        "reference_summary_sha256": _sha256(reference_summary),
        "reference_protocol": reference_payload["protocol"],
        "evaluation_split": "val",
        "images": len(dataset),
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "workers": int(workers),
        "max_det": int(max_det),
        "confidence_threshold": float(confidence),
        "endpoint_tolerance": float(endpoint_tolerance),
        "training_executed": False,
        "test_images_accessed": False,
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="D0FT/AF2 box-score factorial, validation only")
    parser.add_argument("--d0ft-checkpoint", required=True)
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--endpoint-tolerance", type=float, default=0.002)
    args = parser.parse_args()
    result = run_af2_box_score_factorial(
        args.d0ft_checkpoint,
        args.af2_checkpoint,
        args.data_root,
        args.grouped_summary,
        args.reference_summary,
        args.output,
        device=args.device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        max_det=args.max_det,
        confidence=args.confidence,
        endpoint_tolerance=args.endpoint_tolerance,
    )
    headline = {
        arm: {metric: result["results"][arm][metric] for metric in METRICS}
        for arm in ARMS
    }
    print(json.dumps(headline, indent=2, ensure_ascii=False))
    print("CALIBRATION:", json.dumps(result["calibration"], indent=2))
    print("GATES:", result["gates"])
    print("DECISION:", result["decision"])
    print("NEXT:", result["next"])
    print("TRAINING: False | TEST: False")
    print("SUMMARY:", result["summary"])


if __name__ == "__main__":
    main()
