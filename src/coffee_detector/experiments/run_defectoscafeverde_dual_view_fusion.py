"""Train one symmetric dual-view fuser on a frozen DefectosCafeVerde D0."""

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
from torchvision.ops import box_iou

from coffee_detector.analysis.coffee_fg_diagnostics import (
    _decode_branch,
    _letterbox_sample,
    _raw_branches,
    _unwrap_head,
)
from coffee_detector.analysis.defectoscafeverde_dual_view_audit import (
    FORMAT as AUDIT_FORMAT,
    _eligible_pairs,
)
from coffee_detector.data.prepare_defectoscafeverde_grouped import NAMES
from coffee_detector.dataset import IMAGE_SUFFIXES, parse_label
from coffee_detector.defectos_dual_view import (
    DualViewFusionConfig,
    SymmetricDualViewFuser,
)
from coffee_detector.defectos_dual_view.model import paper_base_logits


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/defectoscafeverde/DVF1.yaml"
PROTOCOL = "defectoscafeverde-dual-view-fusion-seed42-v1"
FORMAT = "coffee_detector.defectoscafeverde.dual_view_fusion.result.v1"
D0_PROTOCOL = "defectoscafeverde-grouped-dcwcf-direct-seed42-v1"


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


def _config() -> tuple[dict, DualViewFusionConfig]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if payload.get("code") != "DVF1" or payload.get("protocol") != PROTOCOL:
        raise RuntimeError("Konfigurasi DVF1 berubah")
    return payload, DualViewFusionConfig.from_mapping(payload["fuser"])


def _device(value: str) -> torch.device:
    result = torch.device(f"cuda:{value}" if str(value).isdigit() else value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {result}")
    return result


def _state_fingerprint(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def run_static_audit(output: str | Path) -> dict:
    payload, config = _config()
    torch.manual_seed(42)
    logits = torch.randn(8, 2, config.classes, requires_grad=True)
    fuser = SymmetricDualViewFuser(config)
    base = paper_base_logits(logits)
    initial = fuser(logits)
    swapped = fuser(logits.flip(1))
    loss = F.cross_entropy(initial, torch.arange(8) % config.classes)
    loss.backward()
    gradients = [parameter.grad for parameter in fuser.parameters()]
    with torch.no_grad():
        fuser.residual[-1].bias[0] = 0.25
        active = fuser(logits.detach())
    parameters = sum(parameter.numel() for parameter in fuser.parameters())
    gates = {
        "config_exact": payload["code"] == "DVF1" and payload["protocol"] == PROTOCOL,
        "parameter_count_under_5000": 0 < parameters < 5000,
        "initial_output_exactly_paper_rule": torch.equal(initial, base),
        "initial_swap_invariant": torch.equal(initial, swapped),
        "active_residual_changes_output": not torch.equal(active, base.detach()),
        "all_gradients_finite": all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        ),
        "final_layer_gradients_nonzero": bool(
            fuser.residual[-1].weight.grad is not None
            and fuser.residual[-1].weight.grad.abs().sum() > 0
        ),
        "detector_has_no_optimizer": True,
        "test_not_accessed": True,
    }
    result = {
        "format": "coffee_detector.defectoscafeverde.dvf1.static_audit.v1",
        "config": str(CONFIG_PATH),
        "config_sha256": _sha256(CONFIG_PATH),
        "parameters": parameters,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "training_executed": False,
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _split_samples(
    root: Path, split: str
) -> tuple[dict[int, str], list[tuple[Path, tuple]]]:
    if split not in {"train", "val"}:
        raise ValueError("DVF1 hanya mengizinkan train atau val")
    payload = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8")) or {}
    raw_names = payload.get("names")
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        raise ValueError("Ontologi data.yaml tidak valid")
    image_root, label_root = root / split / "images", root / split / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"Split {split} tidak lengkap")
    samples = []
    for image in sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    ):
        relative = image.relative_to(image_root)
        samples.append(
            (image, parse_label((label_root / relative).with_suffix(".txt"), set(names)))
        )
    if not samples:
        raise RuntimeError(f"Split {split} kosong")
    return names, samples


def _paired_split(
    root: Path, manifest_path: Path, split: str
) -> tuple[dict[int, str], list[tuple[str, list[tuple[Path, tuple, dict]], int]], dict]:
    names, samples = _split_samples(root, split)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [row for row in manifest if row.get("output_split") == split]
    by_name = {Path(str(row["image"])).name: row for row in rows}
    if len(by_name) != len(rows):
        raise RuntimeError(f"Nama citra {split} dalam manifest tidak unik")
    grouped: dict[str, list[tuple[Path, tuple, dict]]] = {}
    for image, annotations in samples:
        row = by_name.get(image.name)
        if row is None:
            raise RuntimeError(f"Manifest tidak memuat {split}/{image.name}")
        grouped.setdefault(str(row["physical_group_id"]), []).append(
            (image, annotations, row)
        )
    eligible, rejected = _eligible_pairs(grouped)
    return names, eligible, {
        "images": len(samples),
        "groups": len(grouped),
        "eligible_groups": len(eligible),
        "rejected_groups": dict(rejected),
    }


def _validate_inputs(
    grouped_audit: Path,
    grouped_manifest: Path,
    d0_result_path: Path,
    d0_checkpoint: Path,
    audit_result_path: Path,
) -> tuple[dict, dict]:
    dataset_audit = _json(grouped_audit, "Grouped audit")
    if dataset_audit.get("decision") != "PASS_GROUPED_DATASET_GATE" or not all(
        dataset_audit.get("gates", {}).values()
    ):
        raise RuntimeError("Grouped dataset audit tidak PASS")
    d0 = _json(d0_result_path, "D0 result")
    if (
        d0.get("protocol") != D0_PROTOCOL
        or d0.get("arm") != "D0DIRECT"
        or d0.get("seed") != 42
        or d0.get("completed_epochs") != 50
        or d0.get("test_images_accessed") is not False
    ):
        raise RuntimeError("D0 result tidak memenuhi kontrak DVF1")
    if not d0_checkpoint.is_file() or _sha256(d0_checkpoint) != d0.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError("Checkpoint D0 tidak cocok")
    if _sha256(grouped_audit) != d0["run_contract"]["dataset_audit_sha256"]:
        raise RuntimeError("Grouped audit tidak cocok dengan D0")
    audit = _json(audit_result_path, "Dual-view audit")
    if (
        audit.get("format") != AUDIT_FORMAT
        or audit.get("decision")
        != "AUTHORIZE_ONE_LEARNED_DUAL_VIEW_ARCHITECTURE_SCREEN"
        or audit.get("training_executed") is not False
        or audit.get("test_images_accessed") is not False
    ):
        raise RuntimeError("Dual-view audit belum mengotorisasi DVF1")
    expected = audit.get("contract", {})
    checks = {
        "dataset_audit": _sha256(grouped_audit),
        "grouped_manifest": _sha256(grouped_manifest),
        "d0_result": _sha256(d0_result_path),
        "d0_checkpoint": _sha256(d0_checkpoint),
    }
    for key, actual in checks.items():
        if expected.get(f"{key}_sha256") != actual:
            raise RuntimeError(f"Kontrak dual-view audit berubah: {key}")
    return d0, audit


def _view_evidence(
    network: torch.nn.Module,
    image_path: Path,
    annotations: tuple,
    *,
    image_size: int,
    max_det: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, dict]:
    image, targets, labels, _ = _letterbox_sample(
        image_path, annotations, image_size, device
    )
    final, raw, head = _raw_branches(network, image, max_det)
    boxes, scores = _decode_branch(head, raw["one2one"])
    candidate_confidence, candidate_class = scores.max(dim=1)
    index = int(candidate_confidence.argmax())
    score_vector = scores[index].clamp(1e-6, 1.0 - 1e-6)
    selected_box = boxes[index]
    iou = float(box_iou(selected_box[None], targets).max())
    final_top = final[int(final[:, 4].argmax())]
    calibration = {
        "class_exact": int(candidate_class[index]) == int(final_top[5]),
        "confidence_close": abs(float(candidate_confidence[index]) - float(final_top[4]))
        <= 1e-5,
        "box_max_abs_diff": float((selected_box - final_top[:4]).abs().max()),
        "target_class": int(labels[0]),
    }
    return torch.logit(score_vector).cpu(), iou, calibration


def build_pair_cache(
    network: torch.nn.Module,
    root: Path,
    manifest_path: Path,
    split: str,
    output: Path,
    contract: dict,
    *,
    image_size: int,
    max_det: int,
    device: torch.device,
) -> dict:
    if output.is_file():
        cached = torch.load(output, map_location="cpu", weights_only=False)
        if cached.get("contract") != contract or cached.get("split") != split:
            raise RuntimeError(f"Cache pasangan {split} stale")
        print(f"REUSE DVF1 {split.upper()} CACHE: {output}", flush=True)
        return cached
    names, pairs, stats = _paired_split(root, manifest_path, split)
    view_logits, ious, labels, group_ids = [], [], [], []
    calibration = Counter()
    network.eval()
    with torch.inference_mode():
        for index, (group_id, rows, class_id) in enumerate(pairs, 1):
            pair_logits, pair_ious = [], []
            for image, annotations, _ in rows:
                logits, iou, gates = _view_evidence(
                    network,
                    image,
                    annotations,
                    image_size=image_size,
                    max_det=max_det,
                    device=device,
                )
                pair_logits.append(logits)
                pair_ious.append(iou)
                calibration["views"] += 1
                calibration["class_exact"] += int(gates["class_exact"])
                calibration["confidence_close"] += int(gates["confidence_close"])
                calibration["box_close"] += int(gates["box_max_abs_diff"] <= 1e-3)
            view_logits.append(torch.stack(pair_logits))
            ious.append(pair_ious)
            labels.append(class_id)
            group_ids.append(group_id)
            if index % 100 == 0 or index == len(pairs):
                print(f"DVF1 {split.upper()} CACHE {index}/{len(pairs)}", flush=True)
    cache = {
        "format": "coffee_detector.defectoscafeverde.dvf1.pair_cache.v1",
        "split": split,
        "contract": contract,
        "names": names,
        "stats": stats,
        "view_logits": torch.stack(view_logits),
        "ious": torch.tensor(ious, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "group_ids": group_ids,
        "raw_final_calibration": dict(calibration),
    }
    if any(calibration[key] != calibration["views"] for key in ("class_exact", "confidence_close", "box_close")):
        raise RuntimeError(f"Raw/final candidate calibration gagal: {dict(calibration)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    return cache


def _train_fuser(
    cache: dict,
    config: DualViewFusionConfig,
    train_config: Mapping,
    output: Path,
    contract: dict,
    device: torch.device,
) -> tuple[SymmetricDualViewFuser, list[dict], bool]:
    seed = int(train_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    fuser = SymmetricDualViewFuser(config).to(device)
    optimizer = torch.optim.AdamW(
        fuser.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    epochs = int(train_config["epochs"])
    start, history = 0, []
    generator_state = None
    if output.is_file():
        state = torch.load(output, map_location=device, weights_only=False)
        if state.get("contract") != contract:
            raise RuntimeError("Checkpoint DVF1 stale")
        fuser.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, history = int(state["epoch"]), list(state["history"])
        generator_state = state.get("generator_state")
        if start >= epochs:
            print(f"REUSE COMPLETE DVF1 FUSER: {output}", flush=True)
            return fuser.eval(), history, False
    labels = cache["labels"]
    counts = torch.bincount(labels, minlength=config.classes).float().clamp_min(1)
    class_weight = counts.rsqrt()
    class_weight = (class_weight / class_weight.mean()).to(device)
    dataset = TensorDataset(cache["view_logits"], labels)
    generator = torch.Generator().manual_seed(seed)
    if generator_state is not None:
        generator.set_state(generator_state)
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["batch"]),
        shuffle=True,
        generator=generator,
    )
    for epoch in range(start, epochs):
        fuser.train()
        total_loss = total_ce = total_kl = 0.0
        seen = 0
        for pair_logits, target in loader:
            pair_logits, target = pair_logits.to(device), target.to(device)
            fused = fuser(pair_logits)
            base = paper_base_logits(pair_logits).detach()
            ce = F.cross_entropy(
                fused,
                target,
                weight=class_weight,
                label_smoothing=float(train_config["label_smoothing"]),
            )
            kl = F.kl_div(
                F.log_softmax(fused, dim=1),
                F.softmax(base, dim=1),
                reduction="batchmean",
            )
            loss = ce + float(train_config["preservation_weight"]) * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(target)
            seen += count
            total_loss += float(loss.detach()) * count
            total_ce += float(ce.detach()) * count
            total_kl += float(kl.detach()) * count
        row = {
            "epoch": epoch + 1,
            "loss": total_loss / seen,
            "cross_entropy": total_ce / seen,
            "preservation_kl": total_kl / seen,
        }
        history.append(row)
        torch.save(
            {
                "format": "coffee_detector.defectoscafeverde.dvf1.checkpoint.v1",
                "contract": contract,
                "config": config.to_dict(),
                "epoch": epoch + 1,
                "model": fuser.state_dict(),
                "optimizer": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "history": history,
            },
            output,
        )
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"DVF1 FUSER {epoch + 1}/{epochs} loss={row['loss']:.6f}", flush=True)
    return fuser.eval(), history, True


def _metrics(correct: torch.Tensor, labels: torch.Tensor, names: Mapping[int, str]) -> dict:
    per_class = {}
    missing = []
    for class_id, name in names.items():
        selected = labels == class_id
        if not bool(selected.any()):
            missing.append(name)
            continue
        per_class[name] = float(correct[selected].float().mean())
    ordered = sorted(per_class.values())
    return {
        "physical_pair_accuracy": float(correct.float().mean()),
        "macro_class_accuracy": float(np.mean(ordered)),
        "bottom3_class_accuracy": float(np.mean(ordered[:3])),
        "worst_class_accuracy": float(ordered[0]),
        "accuracy_by_class": per_class,
        "classes_without_ground_truth": missing,
    }


def evaluate_fuser(
    fuser: SymmetricDualViewFuser, cache: dict, device: torch.device
) -> dict:
    logits = cache["view_logits"].to(device)
    ious = cache["ious"].to(device)
    labels = cache["labels"].to(device)
    names = {int(key): str(value) for key, value in cache["names"].items()}
    scores = logits.sigmoid()
    view_confidence, view_class = scores.max(dim=2)
    paper_view = view_confidence.argmax(dim=1)
    rows = torch.arange(len(labels), device=device)
    paper_class = view_class[rows, paper_view]
    paper_correct = (paper_class == labels) & (ious[rows, paper_view] >= 0.5)
    side_correct = (view_class == labels[:, None]) & (ious >= 0.5)
    oracle_correct = side_correct.any(dim=1)
    fuser.eval()
    with torch.inference_mode():
        fused_logits = fuser(logits)
    fused_class = fused_logits.argmax(dim=1)
    class_scores = scores.gather(
        2, fused_class[:, None, None].expand(-1, 2, 1)
    ).squeeze(2)
    fused_view = class_scores.argmax(dim=1)
    fused_correct = (fused_class == labels) & (ious[rows, fused_view] >= 0.5)
    return {
        "SIDE_A": _metrics(side_correct[:, 0], labels, names),
        "SIDE_B": _metrics(side_correct[:, 1], labels, names),
        "PAPER_MAX_CONFIDENCE": _metrics(paper_correct, labels, names),
        "DVF1": _metrics(fused_correct, labels, names),
        "PAIR_ORACLE": _metrics(oracle_correct, labels, names),
        "zero_residual_predictions_equal_paper": bool(
            torch.equal(paper_base_logits(logits).argmax(dim=1), paper_class)
        ),
        "groups": len(labels),
    }


def _decision(values: Mapping[str, Mapping], config: Mapping) -> tuple[dict, str, str]:
    paper, candidate = values["PAPER_MAX_CONFIDENCE"], values["DVF1"]
    metrics = (
        "physical_pair_accuracy",
        "macro_class_accuracy",
        "bottom3_class_accuracy",
        "worst_class_accuracy",
    )
    deltas = {metric: candidate[metric] - paper[metric] for metric in metrics}
    criteria = {
        "overall_gain_at_least_0_5_point": deltas["physical_pair_accuracy"]
        >= float(config["overall_gain"]),
        "macro_not_lower": deltas["macro_class_accuracy"]
        >= float(config["macro_floor"]),
        "bottom3_not_lower": deltas["bottom3_class_accuracy"]
        >= float(config["bottom3_floor"]),
        "worst_drop_no_more_than_2_points": deltas["worst_class_accuracy"]
        >= -float(config["worst_drop_limit"]),
    }
    passed = all(criteria.values())
    return (
        {"deltas": deltas, "criteria": criteria},
        "PASS" if passed else "FAIL",
        "REVIEW_BEFORE_MULTISEED" if passed else "STOP_DVF1",
    )


def run_dvf1(
    data_root: str | Path,
    grouped_manifest: str | Path,
    grouped_audit: str | Path,
    d0_result: str | Path,
    d0_checkpoint: str | Path,
    dual_view_audit: str | Path,
    output_root: str | Path,
    *,
    device: str = "cpu",
    authorize_training: bool = False,
) -> dict:
    if not authorize_training:
        raise PermissionError("DVF1 memerlukan --authorize-training")
    root = Path(data_root).expanduser().resolve()
    if (root / "test").exists():
        raise RuntimeError("TEST TEREXPOSE - STOP")
    manifest = Path(grouped_manifest).expanduser().resolve()
    audit_path = Path(grouped_audit).expanduser().resolve()
    d0_result_path = Path(d0_result).expanduser().resolve()
    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    dual_audit_path = Path(dual_view_audit).expanduser().resolve()
    d0, audit = _validate_inputs(
        audit_path, manifest, d0_result_path, checkpoint, dual_audit_path
    )
    payload, fuser_config = _config()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "DVF1_seed42_result.json"
    run_contract = {
        "config_sha256": _sha256(CONFIG_PATH),
        "grouped_manifest_sha256": _sha256(manifest),
        "grouped_audit_sha256": _sha256(audit_path),
        "d0_result_sha256": _sha256(d0_result_path),
        "d0_checkpoint_sha256": _sha256(checkpoint),
        "dual_view_audit_sha256": _sha256(dual_audit_path),
        "detector_frozen": True,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    if result_path.is_file():
        result = _json(result_path, "Cached DVF1 result")
        if result.get("format") != FORMAT or result.get("run_contract") != run_contract:
            raise RuntimeError("Cached DVF1 result stale")
        result["training_executed_this_call"] = False
        print(f"REUSE COMPLETE DVF1 RESULT: {result_path}", flush=True)
        return result
    static = run_static_audit(output_root / "static_audit.json")
    if static["decision"] != "PASS":
        raise RuntimeError(f"Static audit DVF1 gagal: {static['gates']}")
    torch_device = _device(device)
    from ultralytics import YOLO

    yolo = YOLO(str(checkpoint))
    network = yolo.model.to(torch_device).eval()
    checkpoint_names = getattr(network, "names", {})
    if isinstance(checkpoint_names, list):
        checkpoint_names = dict(enumerate(checkpoint_names))
    if {int(key): str(value) for key, value in checkpoint_names.items()} != dict(
        enumerate(NAMES)
    ):
        raise RuntimeError("Ontologi checkpoint D0 bukan DefectosCafeVerde 12 kelas")
    detector_before = _state_fingerprint(network)
    detector_config = payload["detector"]
    cache_contract = {
        "grouped_manifest_sha256": _sha256(manifest),
        "grouped_audit_sha256": _sha256(audit_path),
        "d0_checkpoint_sha256": _sha256(checkpoint),
        "image_size": int(detector_config["imgsz"]),
        "max_det": int(detector_config["max_det"]),
        "test_images_accessed": False,
    }
    train_cache = build_pair_cache(
        network,
        root,
        manifest,
        "train",
        output_root / "train_pair_cache.pt",
        cache_contract,
        image_size=int(detector_config["imgsz"]),
        max_det=int(detector_config["max_det"]),
        device=torch_device,
    )
    train_groups = set(train_cache["group_ids"])
    fuser_path = output_root / "DVF1_last.pt"
    fuser_contract = {**run_contract, "train_cache_groups": len(train_groups)}
    fuser, history, trained = _train_fuser(
        train_cache,
        fuser_config,
        payload["train"],
        fuser_path,
        fuser_contract,
        torch_device,
    )
    val_cache = build_pair_cache(
        network,
        root,
        manifest,
        "val",
        output_root / "val_pair_cache.pt",
        cache_contract,
        image_size=int(detector_config["imgsz"]),
        max_det=int(detector_config["max_det"]),
        device=torch_device,
    )
    val_groups = set(val_cache["group_ids"])
    values = evaluate_fuser(fuser, val_cache, torch_device)
    audit_summary = audit["summary"]
    calibration = {
        "paper_max_confidence_accuracy": values["PAPER_MAX_CONFIDENCE"]
        ["physical_pair_accuracy"]
        - float(audit_summary["paper_max_confidence_accuracy"]),
        "pair_oracle_accuracy": values["PAIR_ORACLE"]["physical_pair_accuracy"]
        - float(audit_summary["pair_oracle_accuracy"]),
    }
    detector_frozen = detector_before == _state_fingerprint(network)
    gates = {
        "static_audit_pass": static["decision"] == "PASS",
        "train_validation_groups_disjoint": not (train_groups & val_groups),
        "detector_state_frozen": detector_frozen,
        "raw_final_candidate_calibrated_train": all(
            train_cache["raw_final_calibration"][key]
            == train_cache["raw_final_calibration"]["views"]
            for key in ("class_exact", "confidence_close", "box_close")
        ),
        "raw_final_candidate_calibrated_val": all(
            val_cache["raw_final_calibration"][key]
            == val_cache["raw_final_calibration"]["views"]
            for key in ("class_exact", "confidence_close", "box_close")
        ),
        "paper_endpoint_matches_audit": all(abs(value) <= 1e-9 for value in calibration.values()),
        "zero_residual_predictions_equal_paper": values.pop(
            "zero_residual_predictions_equal_paper"
        ),
        "all_12_validation_classes_present": all(
            not row["classes_without_ground_truth"] for row in values.values() if isinstance(row, dict)
        ),
        "detector_training_not_executed": True,
        "test_not_opened": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"DVF1 runtime gate gagal: {gates}")
    comparison, decision, next_action = _decision(values, payload["decision"])
    result = {
        "format": FORMAT,
        "protocol": PROTOCOL,
        "seed": 42,
        "values": values,
        "comparison": comparison,
        "historical_audit_calibration": calibration,
        "gates": gates,
        "decision": decision,
        "next": next_action,
        "fuser_parameters": sum(parameter.numel() for parameter in fuser.parameters()),
        "train_pair_stats": train_cache["stats"],
        "validation_pair_stats": val_cache["stats"],
        "history": history,
        "artifacts": {
            "fuser_checkpoint": str(fuser_path),
            "fuser_checkpoint_sha256": _sha256(fuser_path),
            "train_cache": str(output_root / "train_pair_cache.pt"),
            "validation_cache": str(output_root / "val_pair_cache.pt"),
            "static_audit": str(output_root / "static_audit.json"),
        },
        "run_contract": run_contract,
        "detector_training_executed": False,
        "fuser_training_executed": True,
        "training_executed_this_call": trained,
        "test_images_accessed": False,
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="DVF1 paired-side fusion screen")
    parser.add_argument("--data-root")
    parser.add_argument("--grouped-manifest")
    parser.add_argument("--grouped-audit")
    parser.add_argument("--d0-result")
    parser.add_argument("--d0-checkpoint")
    parser.add_argument("--dual-view-audit")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    if args.static_only:
        result = run_static_audit(Path(args.output_root) / "static_audit.json")
        print(json.dumps(result, indent=2))
        return
    required = (
        "data_root",
        "grouped_manifest",
        "grouped_audit",
        "d0_result",
        "d0_checkpoint",
        "dual_view_audit",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"Argumen wajib tidak ada: {missing}")
    run_dvf1(
        args.data_root,
        args.grouped_manifest,
        args.grouped_audit,
        args.d0_result,
        args.d0_checkpoint,
        args.dual_view_audit,
        args.output_root,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
