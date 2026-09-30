"""Train and evaluate FSRC1 on a frozen DefectosCafeVerde D0 detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from coffee_detector.analysis.af2_box_score_factorial import (
    _match_predictions,
    _network_raw,
    summarize_detection_stats,
)
from coffee_detector.data.prepare_defectoscafeverde_grouped import NAMES
from coffee_detector.dataset import discover_layout
from coffee_detector.defectos_fsrc import (
    FSRCConfig,
    SpectralReliabilityCalibrator,
    apply_reliability_suppression,
    candidate_features,
    stationary_haar_energy,
)
from coffee_detector.experiments.run_defectoscafeverde_af2_direct import (
    validate_development_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/defectoscafeverde/FSRC1.yaml"
PROTOCOL = "defectoscafeverde-fsrc1-frozen-d0-seed42-v1"
FORMAT = "coffee_detector.defectoscafeverde.fsrc1.result.v1"
D0_PROTOCOL = "defectoscafeverde-grouped-dcwcf-direct-seed42-v1"
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
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


def _config() -> tuple[dict, FSRCConfig]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("code") != "FSRC1":
        raise RuntimeError("Konfigurasi FSRC1 berubah")
    return payload, FSRCConfig.from_mapping(payload["fsrc"])


def _device(value: str) -> torch.device:
    device = torch.device(f"cuda:{value}" if str(value).isdigit() else value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {device}")
    return device


def _split_loader(
    yolo_model,
    yaml_path: Path,
    split: str,
    *,
    image_size: int,
    batch_size: int,
    workers: int,
):
    """Build a deterministic, augmentation-free loader for train or val."""

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    if split not in {"train", "val"}:
        raise ValueError("FSRC hanya mengizinkan train atau val")
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
        data[split],
        int(batch_size),
        data,
        mode="val",
        stride=32,
    )
    loader = build_dataloader(
        dataset,
        batch=int(batch_size),
        workers=int(workers),
        shuffle=False,
        rank=-1,
    )
    return dataset, loader


def _validate_d0(
    d0_result: str | Path,
    d0_checkpoint: str | Path,
    grouped_audit: str | Path,
) -> tuple[dict, Path]:
    result = _json(d0_result, "D0 result")
    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"D0 checkpoint: {checkpoint}")
    if (
        result.get("protocol") != D0_PROTOCOL
        or result.get("arm") != "D0DIRECT"
        or result.get("seed") != 42
        or result.get("completed_epochs") != 50
        or result.get("evaluation_split") != "val"
        or result.get("test_images_accessed") is not False
    ):
        raise RuntimeError("D0 result tidak memenuhi kontrak FSRC1")
    if _sha256(checkpoint) != result.get("checkpoint_sha256"):
        raise RuntimeError("SHA checkpoint D0 tidak cocok")
    if _sha256(grouped_audit) != result["run_contract"]["dataset_audit_sha256"]:
        raise RuntimeError("Grouped audit tidak cocok dengan D0")
    return result, checkpoint


def run_fsrc_static_audit(output: str | Path) -> dict:
    payload, config = _config()
    torch.manual_seed(42)
    image = torch.rand(2, 3, 64, 64)
    predictions = [
        torch.tensor(
            [
                [4.0, 4.0, 28.0, 28.0, 0.90, 9.0],
                [5.0, 5.0, 27.0, 27.0, 0.70, 9.0],
            ]
        ),
        torch.tensor([[20.0, 20.0, 50.0, 50.0, 0.80, 3.0]]),
    ]
    cue = stationary_haar_energy(image, levels=config.levels)
    features, _ = candidate_features(image, predictions, config)
    calibrator = SpectralReliabilityCalibrator(config)
    reliability = calibrator.reliability(features)
    active = apply_reliability_suppression(predictions, reliability, config)
    disabled = apply_reliability_suppression(
        predictions, reliability, config, enabled=False
    )
    added_parameters = sum(parameter.numel() for parameter in calibrator.parameters())
    gates = {
        "config_code_exact": payload["code"] == "FSRC1",
        "fixed_two_level_stationary_haar": cue.shape == (2, 6, 64, 64),
        "wavelet_finite": bool(torch.isfinite(cue).all()),
        "wavelet_parameter_free": True,
        "feature_dimension_exact": features.shape == (3, config.feature_dimension),
        "calibrator_parameters_under_2000": 0 < added_parameters < 2000,
        "calibrator_off_exact_identity": all(
            torch.equal(left, right) for left, right in zip(predictions, disabled)
        ),
        "active_boxes_exact": all(
            torch.equal(left[:, :4], right[:, :4])
            for left, right in zip(predictions, active)
        ),
        "active_classes_exact": all(
            torch.equal(left[:, 5], right[:, 5])
            for left, right in zip(predictions, active)
        ),
        "active_scores_never_increase": all(
            bool((right[:, 4] <= left[:, 4]).all())
            for left, right in zip(predictions, active)
        ),
        "active_scores_bounded_by_floor": all(
            bool((right[:, 4] >= left[:, 4] * config.suppression_floor).all())
            for left, right in zip(predictions, active)
        ),
        "test_not_accessed": True,
    }
    result = {
        "format": "coffee_detector.defectoscafeverde.fsrc1.static_audit.v1",
        "config": str(CONFIG_PATH),
        "config_sha256": _sha256(CONFIG_PATH),
        "feature_dimension": config.feature_dimension,
        "added_parameters": added_parameters,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "training_executed": False,
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _prediction_targets(
    batch: Mapping[str, torch.Tensor],
    sample_index: int,
    image_height: int,
    image_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from ultralytics.utils import ops

    mask = batch["batch_idx"].to(device) == sample_index
    classes = batch["cls"].to(device).long().reshape(-1)[mask]
    boxes = batch["bboxes"].to(device)[mask]
    if len(boxes):
        boxes = ops.xywh2xyxy(boxes) * boxes.new_tensor(
            [image_width, image_height, image_width, image_height]
        )
    return boxes, classes


def _cache_contract(
    d0_checkpoint: Path,
    grouped_audit: Path,
    config: FSRCConfig,
) -> dict[str, Any]:
    return {
        "format": "coffee_detector.defectoscafeverde.fsrc1.train_cache.v1",
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "grouped_audit_sha256": _sha256(grouped_audit),
        "config_sha256": _sha256(CONFIG_PATH),
        "feature_dimension": config.feature_dimension,
        "split": "train",
        "test_images_accessed": False,
    }


def build_train_cache(
    yolo_model,
    network: torch.nn.Module,
    data_root: Path,
    d0_checkpoint: Path,
    grouped_audit: Path,
    output: Path,
    config: FSRCConfig,
    *,
    device: torch.device,
    image_size: int,
    batch_size: int,
    workers: int,
    max_det: int,
    confidence: float,
) -> dict[str, torch.Tensor]:
    expected = _cache_contract(d0_checkpoint, grouped_audit, config)
    if output.is_file():
        cached = torch.load(output, map_location="cpu", weights_only=False)
        if cached.get("contract") != expected:
            raise RuntimeError("Cache train FSRC stale atau tidak kompatibel")
        print(f"REUSE FSRC TRAIN CACHE: {output}", flush=True)
        return cached
    dataset, loader = _split_loader(
        yolo_model,
        data_root / "data.yaml",
        "train",
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
    )
    rows = {key: [] for key in ("features", "score", "label", "class_id")}
    thresholds = torch.tensor([0.5], device=device)
    completed = 0
    network.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True).float().div_(255.0)
            final = _network_raw(network, images, max_det)[0]
            predictions = [row[row[:, 4] > confidence] for row in final]
            features, _ = candidate_features(images, predictions, config)
            labels = []
            scores = []
            classes = []
            height, width = images.shape[-2:]
            for sample_index, prediction in enumerate(predictions):
                target_boxes, target_classes = _prediction_targets(
                    batch, sample_index, height, width, device
                )
                predicted_classes = prediction[:, 5].long()
                correct = _match_predictions(
                    predicted_classes,
                    target_classes,
                    prediction[:, :4],
                    target_boxes,
                    thresholds,
                )
                labels.append(correct[:, 0].to(torch.float32))
                scores.append(prediction[:, 4].float().cpu())
                classes.append(predicted_classes.cpu())
            rows["features"].append(features.float().cpu())
            rows["score"].append(torch.cat(scores))
            rows["label"].append(torch.cat(labels))
            rows["class_id"].append(torch.cat(classes))
            completed += len(images)
            if completed % 200 < len(images) or completed == len(dataset):
                print(f"FSRC CACHE TRAIN {completed}/{len(dataset)}", flush=True)
    payload = {
        "contract": expected,
        **{key: torch.cat(value, dim=0) for key, value in rows.items()},
        "images": len(dataset),
    }
    if (
        len(payload["features"]) == 0
        or payload["features"].shape[1] != config.feature_dimension
        or not bool(torch.isfinite(payload["features"]).all())
    ):
        raise RuntimeError("Cache train FSRC tidak valid")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return payload


def _balanced_weights(labels: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    weights = torch.zeros_like(labels)
    for class_id in range(len(NAMES)):
        for label in (0.0, 1.0):
            mask = (classes == class_id) & (labels == label)
            count = int(mask.sum())
            if count:
                weights[mask] = 1.0 / count
    return weights / weights.mean().clamp_min(1e-8)


def _pairwise_ranking_loss(
    adjusted_scores: torch.Tensor,
    labels: torch.Tensor,
    classes: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for class_id in classes.unique().tolist():
        same = classes == int(class_id)
        positives = torch.logit(
            adjusted_scores[same & (labels == 1)].clamp(1e-6, 1 - 1e-6)
        )[:32]
        negatives = torch.logit(
            adjusted_scores[same & (labels == 0)].clamp(1e-6, 1 - 1e-6)
        )[:32]
        if len(positives) and len(negatives):
            losses.append(F.softplus(negatives[:, None] - positives[None, :]).mean())
    return torch.stack(losses).mean() if losses else adjusted_scores.sum() * 0.0


def train_calibrator(
    cache: Mapping[str, Any],
    config: FSRCConfig,
    train_config: Mapping[str, Any],
    output: Path,
    *,
    device: torch.device,
) -> tuple[SpectralReliabilityCalibrator, list[dict[str, float]]]:
    seed = int(train_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    features = cache["features"].float()
    scores = cache["score"].float()
    labels = cache["label"].float()
    classes = cache["class_id"].long()
    weights = _balanced_weights(labels, classes)
    dataset = TensorDataset(features, scores, labels, classes, weights)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["batch"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    calibrator = SpectralReliabilityCalibrator(config).to(device)
    if output.is_file():
        saved = torch.load(output, map_location="cpu", weights_only=False)
        if (
            saved.get("format")
            != "coffee_detector.defectoscafeverde.fsrc1.calibrator.v1"
            or saved.get("config") != config.__dict__
            or saved.get("train_config") != dict(train_config)
            or saved.get("test_images_accessed") is not False
        ):
            raise RuntimeError("Checkpoint calibrator FSRC1 stale atau tidak kompatibel")
        calibrator.load_state_dict(saved["state_dict"])
        print(f"REUSE FSRC CALIBRATOR: {output}", flush=True)
        return calibrator.eval(), list(saved["history"])
    optimizer = torch.optim.AdamW(
        calibrator.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    history = []
    for epoch in range(1, int(train_config["epochs"]) + 1):
        calibrator.train()
        totals = Counter()
        batches = 0
        for batch in loader:
            feature, score, label, class_id, weight = [value.to(device) for value in batch]
            logit = calibrator(feature)
            reliability = logit.sigmoid()
            factor = config.suppression_floor + (
                1.0 - config.suppression_floor
            ) * reliability
            adjusted = score * factor
            bce = (
                F.binary_cross_entropy_with_logits(logit, label, reduction="none")
                * weight
            ).mean()
            ranking = _pairwise_ranking_loss(adjusted, label, class_id)
            positive = label == 1
            preservation = (
                (1.0 - reliability[positive]).square().mean()
                if bool(positive.any())
                else reliability.sum() * 0.0
            )
            loss = (
                bce
                + float(train_config["ranking_weight"]) * ranking
                + float(train_config["preservation_weight"]) * preservation
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 5.0)
            optimizer.step()
            totals.update(
                loss=float(loss.detach()),
                bce=float(bce.detach()),
                ranking=float(ranking.detach()),
                preservation=float(preservation.detach()),
            )
            batches += 1
        row = {key: value / max(batches, 1) for key, value in totals.items()}
        row["epoch"] = epoch
        history.append(row)
        print(
            f"FSRC CALIBRATOR {epoch}/{train_config['epochs']} "
            f"loss={row['loss']:.5f}",
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "coffee_detector.defectoscafeverde.fsrc1.calibrator.v1",
            "state_dict": calibrator.state_dict(),
            "config": config.__dict__,
            "train_config": dict(train_config),
            "history": history,
            "test_images_accessed": False,
        },
        output,
    )
    return calibrator.eval(), history


def _empty_stats() -> dict[str, list[np.ndarray]]:
    return {
        key: [] for key in ("tp", "confidence", "predicted_class", "target_class")
    }


def _finalize(stats: Mapping[str, list[np.ndarray]], names: Mapping[int, str]) -> dict:
    return summarize_detection_stats(
        np.concatenate(stats["tp"], axis=0),
        np.concatenate(stats["confidence"], axis=0),
        np.concatenate(stats["predicted_class"], axis=0),
        np.concatenate(stats["target_class"], axis=0),
        names,
    )


def evaluate_pair(
    yolo_model,
    network: torch.nn.Module,
    calibrator: SpectralReliabilityCalibrator,
    data_root: Path,
    config: FSRCConfig,
    *,
    device: torch.device,
    image_size: int,
    batch_size: int,
    workers: int,
    max_det: int,
    confidence: float,
) -> tuple[dict, dict[str, bool]]:
    dataset, loader = _split_loader(
        yolo_model,
        data_root / "data.yaml",
        "val",
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
    )
    names = dict(enumerate(NAMES))
    stats = {name: _empty_stats() for name in ("D0DIRECT", "FSRC1")}
    thresholds = torch.linspace(0.5, 0.95, 10, device=device)
    exact_boxes = True
    exact_classes = True
    scores_never_increase = True
    completed = 0
    network.eval()
    calibrator.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True).float().div_(255.0)
            final = _network_raw(network, images, max_det)[0]
            baseline = [row[row[:, 4] > confidence] for row in final]
            features, _ = candidate_features(images, baseline, config)
            reliability = calibrator.reliability(features)
            candidate = apply_reliability_suppression(baseline, reliability, config)
            for left, right in zip(baseline, candidate):
                exact_boxes &= torch.equal(left[:, :4], right[:, :4])
                exact_classes &= torch.equal(left[:, 5], right[:, 5])
                scores_never_increase &= bool((right[:, 4] <= left[:, 4]).all())
            height, width = images.shape[-2:]
            for sample_index in range(len(images)):
                target_boxes, target_classes = _prediction_targets(
                    batch, sample_index, height, width, device
                )
                endpoints = {
                    "D0DIRECT": baseline[sample_index],
                    "FSRC1": candidate[sample_index][
                        candidate[sample_index][:, 4] > confidence
                    ],
                }
                for name, prediction in endpoints.items():
                    predicted_classes = prediction[:, 5].long()
                    correct = _match_predictions(
                        predicted_classes,
                        target_classes,
                        prediction[:, :4],
                        target_boxes,
                        thresholds,
                    )
                    stats[name]["tp"].append(correct.numpy())
                    stats[name]["confidence"].append(
                        prediction[:, 4].float().cpu().numpy()
                    )
                    stats[name]["predicted_class"].append(
                        predicted_classes.cpu().numpy()
                    )
                    stats[name]["target_class"].append(target_classes.cpu().numpy())
            completed += len(images)
            if completed % 100 < len(images) or completed == len(dataset):
                print(f"FSRC VALIDATION {completed}/{len(dataset)}", flush=True)
    return (
        {name: _finalize(value, names) for name, value in stats.items()},
        {
            "boxes_exact": exact_boxes,
            "classes_exact": exact_classes,
            "scores_never_increase": scores_never_increase,
        },
    )


def _decision(results: Mapping[str, Mapping[str, float]]) -> tuple[dict, str, str]:
    deltas = {
        metric: float(results["FSRC1"][metric]) - float(results["D0DIRECT"][metric])
        for metric in METRICS
    }
    criteria = {
        "strong_macro_route": (
            deltas["macro_map50_95"] >= 0.002
            and deltas["bottom3_class_map50_95"] >= -0.001
            and deltas["worst_class_map50_95"] >= -0.001
        ),
        "lower_tail_route": (
            deltas["macro_map50_95"] >= -0.001
            and deltas["bottom3_class_map50_95"] >= 0.005
            and deltas["worst_class_map50_95"] >= 0.0
        ),
        "strict_pareto_improvement": all(value >= 0.0 for value in deltas.values())
        and any(value > 0.0 for value in deltas.values()),
    }
    if criteria["strong_macro_route"] or criteria["lower_tail_route"]:
        return {"deltas": deltas, "criteria": criteria}, "PASS", "REVIEW_BEFORE_MULTISEED"
    if criteria["strict_pareto_improvement"]:
        return {"deltas": deltas, "criteria": criteria}, "RETAIN_EXPLORATORY", "ANALYZE_EFFECT_SIZE"
    if any(value > 0.0 for value in deltas.values()):
        return {"deltas": deltas, "criteria": criteria}, "PARETO_TRADEOFF", "REVIEW_CLASSWISE"
    return {"deltas": deltas, "criteria": criteria}, "FAIL", "STOP_FSRC1"


def run_fsrc1(
    data_root: str | Path,
    grouped_audit: str | Path,
    d0_result: str | Path,
    d0_checkpoint: str | Path,
    output_root: str | Path,
    *,
    device: str = "cpu",
    authorize_training: bool = False,
) -> dict:
    if not authorize_training:
        raise PermissionError("Training calibrator memerlukan --authorize-training")
    data_root = Path(data_root).expanduser().resolve()
    grouped_audit = Path(grouped_audit).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("TEST TEREXPOSE — STOP")
    dataset_contract = validate_development_dataset(data_root, grouped_audit)
    d0_payload, checkpoint = _validate_d0(d0_result, d0_checkpoint, grouped_audit)
    payload, config = _config()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "FSRC1_seed42_result.json"
    if destination.is_file():
        old = _json(destination, "Cached FSRC1 result")
        expected_contract = {
            "d0_checkpoint_sha256": _sha256(checkpoint),
            "d0_result_sha256": _sha256(d0_result),
            "grouped_audit_sha256": _sha256(grouped_audit),
            "config_sha256": _sha256(CONFIG_PATH),
            "base_detector_frozen": True,
            "evaluation_split": "val",
            "test_images_accessed": False,
        }
        if (
            old.get("format") != FORMAT
            or old.get("protocol") != PROTOCOL
            or old.get("run_contract") != expected_contract
            or old.get("test_images_accessed") is not False
        ):
            raise RuntimeError("Cached FSRC1 result stale atau tidak kompatibel")
        old["training_executed_this_call"] = False
        old["summary"] = str(destination)
        print(f"REUSE COMPLETE FSRC1 RESULT: {destination}", flush=True)
        return old
    static_path = output_root / "static_audit.json"
    static = run_fsrc_static_audit(static_path)
    if static["decision"] != "PASS":
        raise RuntimeError("Static audit FSRC1 gagal")
    torch_device = _device(device)
    from ultralytics import YOLO

    yolo_model = YOLO(str(checkpoint))
    network = yolo_model.model.to(torch_device).eval()
    network_names = network.names
    if isinstance(network_names, list):
        network_names = dict(enumerate(network_names))
    if dict(enumerate(NAMES)) != {int(k): str(v) for k, v in network_names.items()}:
        raise RuntimeError("Ontologi checkpoint D0 berubah")
    detector_state_before = {
        key: value.detach().cpu().clone() for key, value in network.state_dict().items()
    }
    detector_cfg = payload["detector"]
    cache = build_train_cache(
        yolo_model,
        network,
        data_root,
        checkpoint,
        grouped_audit,
        output_root / "train_candidate_cache.pt",
        config,
        device=torch_device,
        image_size=int(detector_cfg["imgsz"]),
        batch_size=int(detector_cfg["batch"]),
        workers=int(detector_cfg["workers"]),
        max_det=int(detector_cfg["max_det"]),
        confidence=float(detector_cfg["confidence"]),
    )
    calibrator_path = output_root / "FSRC1_calibrator.pt"
    calibrator_existed = calibrator_path.is_file()
    calibrator, history = train_calibrator(
        cache,
        config,
        payload["train"],
        calibrator_path,
        device=torch_device,
    )
    detector_unchanged = all(
        torch.equal(detector_state_before[key], value.detach().cpu())
        for key, value in network.state_dict().items()
    )
    results, inference_gates = evaluate_pair(
        yolo_model,
        network,
        calibrator,
        data_root,
        config,
        device=torch_device,
        image_size=int(detector_cfg["imgsz"]),
        batch_size=int(detector_cfg["batch"]),
        workers=int(detector_cfg["workers"]),
        max_det=int(detector_cfg["max_det"]),
        confidence=float(detector_cfg["confidence"]),
    )
    calibration = {
        metric: float(results["D0DIRECT"][metric]) - float(d0_payload["metrics"][metric])
        for metric in METRICS
    }
    gates = {
        "dataset_contract_pass": all(dataset_contract["gates"].values()),
        "static_audit_pass": static["decision"] == "PASS",
        "d0_checkpoint_frozen": detector_unchanged,
        "d0_endpoint_calibrated": all(abs(value) <= 0.002 for value in calibration.values()),
        "boxes_exact": inference_gates["boxes_exact"],
        "classes_exact": inference_gates["classes_exact"],
        "scores_never_increase": inference_gates["scores_never_increase"],
        "all_12_validation_classes_present": all(
            not result["classes_without_ground_truth"] for result in results.values()
        ),
        "detector_training_executed": False,
        "test_not_opened": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"FSRC1 evaluator/contract gate gagal: {gates}")
    comparison, decision, next_action = _decision(results)
    class_deltas = {
        name: float(results["FSRC1"]["map50_95_by_class"][name])
        - float(results["D0DIRECT"]["map50_95_by_class"][name])
        for name in NAMES
    }
    result = {
        "format": FORMAT,
        "protocol": PROTOCOL,
        "seed": 42,
        "values": {
            name: {metric: results[name][metric] for metric in METRICS}
            for name in results
        },
        "full_results": results,
        "comparison": comparison,
        "class_deltas": class_deltas,
        "historical_calibration": calibration,
        "gates": gates,
        "decision": decision,
        "next": next_action,
        "calibrator_parameters": sum(
            parameter.numel() for parameter in calibrator.parameters()
        ),
        "train_candidates": len(cache["features"]),
        "train_positive_rate": float(cache["label"].float().mean()),
        "history": history,
        "artifacts": {
            "calibrator": str(calibrator_path),
            "calibrator_sha256": _sha256(calibrator_path),
            "train_cache": str(output_root / "train_candidate_cache.pt"),
            "static_audit": str(static_path),
        },
        "run_contract": {
            "d0_checkpoint_sha256": _sha256(checkpoint),
            "d0_result_sha256": _sha256(d0_result),
            "grouped_audit_sha256": _sha256(grouped_audit),
            "config_sha256": _sha256(CONFIG_PATH),
            "base_detector_frozen": True,
            "evaluation_split": "val",
            "test_images_accessed": False,
        },
        "calibrator_training_executed": True,
        "training_executed_this_call": not calibrator_existed,
        "detector_training_executed": False,
        "test_images_accessed": False,
    }
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    result["summary"] = str(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-D0 FSRC1 seed-42 screen")
    parser.add_argument("--data-root")
    parser.add_argument("--grouped-audit")
    parser.add_argument("--d0-result")
    parser.add_argument("--d0-checkpoint")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    if args.static_only:
        result = run_fsrc_static_audit(Path(args.output_root) / "static_audit.json")
        print(json.dumps(result, indent=2))
        return
    missing = [
        name
        for name in ("data_root", "grouped_audit", "d0_result", "d0_checkpoint")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(f"Argumen wajib untuk run penuh tidak ada: {missing}")
    result = run_fsrc1(
        args.data_root,
        args.grouped_audit,
        args.d0_result,
        args.d0_checkpoint,
        args.output_root,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print("VALUES:", result["values"])
    print("DELTAS:", result["comparison"]["deltas"])
    print("CRITERIA:", result["comparison"]["criteria"])
    print("GATES:", result["gates"])
    print("DECISION:", result["decision"], "| NEXT:", result["next"])
    print("DETECTOR TRAINING: False | CALIBRATOR TRAINING: True | TEST: False")
    print("SUMMARY:", result["summary"])


if __name__ == "__main__":
    main()
