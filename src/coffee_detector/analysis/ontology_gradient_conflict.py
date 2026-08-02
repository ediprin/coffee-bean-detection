from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable

import numpy as np
import torch

from coffee_detector.dataset import discover_layout
from coffee_detector.ontology_marginal import OntologyMarginalConfig, OntologyMarginalizer
from coffee_detector.sni21_ontology import SNI21_CLASSES


DEFAULT_ONTOLOGY_CONFIG = OntologyMarginalConfig(
    mode="semantic",
    tasks=(
        "entity_family",
        "primary_condition",
        "hole_count",
        "integrity_fraction",
        "surface_extent",
    ),
    auxiliary_gain=0.20,
    task_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
)


def summarize_cosines(values: Iterable[float]) -> dict[str, float | int]:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    if not cleaned:
        raise ValueError("Tidak ada cosine gradient yang finite")
    array = np.asarray(cleaned, dtype=np.float64)
    return {
        "batches": int(array.size),
        "mean_cosine": float(fmean(cleaned)),
        "median_cosine": float(median(cleaned)),
        "q25_cosine": float(np.quantile(array, 0.25)),
        "q75_cosine": float(np.quantile(array, 0.75)),
        "negative_batches": int(np.sum(array < 0.0)),
        "negative_fraction": float(np.mean(array < 0.0)),
    }


def conflict_gate(summary: dict[str, Any]) -> bool:
    return bool(
        float(summary["negative_fraction"]) >= 0.50
        and float(summary["median_cosine"]) < 0.0
    )


def route_conflict_decision(
    feature_extractor: dict[str, Any], classification_head: dict[str, Any]
) -> dict[str, Any]:
    trunk_conflict = conflict_gate(feature_extractor)
    head_conflict = conflict_gate(classification_head)
    if trunk_conflict and head_conflict:
        action = "AUTHORIZE_DUAL_HEAD_WITH_SHARED_GRADIENT_PROJECTION"
    elif head_conflict:
        action = "AUTHORIZE_DUAL_HEAD_ISOLATION_ONLY"
    elif trunk_conflict:
        action = "AUTHORIZE_SHARED_GRADIENT_PROJECTION_ONLY"
    else:
        action = "STOP_CONFLICT_AWARE_DIRECTION"
    return {
        "decision": "PASS" if trunk_conflict or head_conflict else "FAIL",
        "feature_extractor_conflict": trunk_conflict,
        "classification_head_conflict": head_conflict,
        "next_action": action,
        "thresholds": {
            "minimum_negative_fraction": 0.50,
            "maximum_median_cosine": 0.0,
        },
        "training_authorized": False,
    }


def _make_audit_loss(model: torch.nn.Module, config: OntologyMarginalConfig):
    from ultralytics.utils.loss import v8DetectionLoss

    class AuditDetectionLoss(v8DetectionLoss):
        def __init__(self, tal_topk: int = 10, tal_topk2: int | None = None):
            super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
            self.ontology = OntologyMarginalizer(config).to(self.device)
            self.last_flat_classification: torch.Tensor | None = None
            self.last_ontology: torch.Tensor | None = None
            self.last_foreground = 0

        def get_assigned_targets_and_loss(self, preds, batch):
            assignments, loss, detached = super().get_assigned_targets_and_loss(
                preds, batch
            )
            fg_mask, target_gt_idx = assignments[:2]
            self.last_flat_classification = loss[1]
            self.last_foreground = int(fg_mask.sum().detach().cpu())
            pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
            auxiliary = pred_scores.sum() * 0.0
            if bool(fg_mask.any()):
                batch_size = pred_scores.shape[0]
                image_size = (
                    torch.tensor(
                        preds["feats"][0].shape[2:],
                        device=self.device,
                        dtype=pred_scores.dtype,
                    )
                    * self.stride[0]
                )
                targets = torch.cat(
                    (
                        batch["batch_idx"].view(-1, 1),
                        batch["cls"].view(-1, 1),
                        batch["bboxes"],
                    ),
                    1,
                )
                targets = self.preprocess(
                    targets.to(self.device),
                    batch_size,
                    scale_tensor=image_size[[1, 0, 1, 0]],
                )
                gt_labels = targets[..., 0].long()
                assigned_labels = gt_labels.gather(1, target_gt_idx.long())
                auxiliary, _ = self.ontology(
                    pred_scores[fg_mask], assigned_labels[fg_mask]
                )
            self.last_ontology = auxiliary
            return assignments, loss, detached

    return AuditDetectionLoss


def _gradient_geometry(
    flat_loss: torch.Tensor,
    ontology_loss: torch.Tensor,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    groups: dict[str, set[str]],
) -> dict[str, dict[str, float | int]]:
    parameters = [parameter for _, parameter in named_parameters]
    flat_gradients = torch.autograd.grad(
        flat_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    ontology_gradients = torch.autograd.grad(
        ontology_loss,
        parameters,
        retain_graph=False,
        allow_unused=True,
    )
    output: dict[str, dict[str, float | int]] = {}
    for group_name, selected_names in groups.items():
        dot = torch.zeros((), device=flat_loss.device, dtype=torch.float64)
        flat_sq = torch.zeros_like(dot)
        ontology_sq = torch.zeros_like(dot)
        shared_tensors = 0
        shared_parameters = 0
        for (name, parameter), flat_gradient, ontology_gradient in zip(
            named_parameters, flat_gradients, ontology_gradients
        ):
            if name not in selected_names or flat_gradient is None or ontology_gradient is None:
                continue
            flat = flat_gradient.detach().to(dtype=torch.float64)
            semantic = ontology_gradient.detach().to(dtype=torch.float64)
            dot = dot + torch.sum(flat * semantic)
            flat_sq = flat_sq + torch.sum(flat * flat)
            ontology_sq = ontology_sq + torch.sum(semantic * semantic)
            shared_tensors += 1
            shared_parameters += parameter.numel()
        flat_norm = torch.sqrt(flat_sq)
        ontology_norm = torch.sqrt(ontology_sq)
        denominator = flat_norm * ontology_norm
        if shared_tensors == 0 or float(denominator) == 0.0:
            raise RuntimeError(f"Gradien bersama kosong pada group {group_name}")
        output[group_name] = {
            "cosine": float((dot / denominator).cpu()),
            "dot_product": float(dot.cpu()),
            "flat_norm": float(flat_norm.cpu()),
            "ontology_norm": float(ontology_norm.cpu()),
            "shared_parameter_tensors": shared_tensors,
            "shared_parameters": shared_parameters,
        }
    return output


def _parameter_groups(model: torch.nn.Module) -> tuple[list, dict[str, set[str]]]:
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    detect_prefix = f"model.{len(model.model) - 1}."
    classification = {
        name
        for name, _ in named
        if name.startswith(detect_prefix)
        and (".cv3." in name or ".one2one_cv3." in name)
    }
    feature_extractor = {
        name for name, _ in named if not name.startswith(detect_prefix)
    }
    return named, {
        "all_shared": {name for name, _ in named},
        "feature_extractor": feature_extractor,
        "classification_head": classification,
    }


def _build_train_loader(data_root: Path, batch_size: int, seed: int):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    layout = discover_layout(data_root)
    if "test" in layout.splits:
        raise RuntimeError("Audit gradient menolak dataset yang mengekspos test")
    data = check_det_dataset(str(layout.yaml_path))
    ordered_names = tuple(str(data["names"][index]) for index in range(data["nc"]))
    if ordered_names != SNI21_CLASSES:
        raise ValueError("Urutan 21 kelas dataset tidak sama dengan ontologi SNI-21")
    cfg = get_cfg(
        overrides={
            "task": "detect",
            "mode": "val",
            "imgsz": 640,
            "batch": batch_size,
            "workers": 2,
            "rect": False,
            "augment": False,
            "seed": seed,
            "deterministic": True,
            "cache": False,
        }
    )
    dataset = build_yolo_dataset(
        cfg,
        str(data["train"]),
        batch_size,
        data,
        mode="val",
        rect=False,
        stride=32,
    )
    loader = build_dataloader(
        dataset,
        batch=batch_size,
        workers=2,
        shuffle=True,
        rank=-1,
        drop_last=True,
    )
    # Ultralytics deliberately creates a private loader generator. Override its
    # fixed library seed so the sampling provenance follows this protocol.
    loader.generator.manual_seed(seed)
    return loader, layout


def run_ontology_gradient_conflict_audit(
    checkpoint: str | Path,
    data_root: str | Path,
    output: str | Path,
    *,
    device: str = "0",
    batch_size: int = 8,
    max_batches: int = 24,
    seed: int = 42,
) -> dict[str, Any]:
    from torch.nn.modules.batchnorm import _BatchNorm
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import E2ELoss

    if batch_size <= 0 or max_batches <= 0:
        raise ValueError("batch_size dan max_batches harus positif")
    data_root = Path(data_root).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    loader, layout = _build_train_loader(data_root, batch_size, seed)
    if device == "cpu" or str(device).startswith("cuda"):
        torch_device = torch.device(device)
    else:
        torch_device = torch.device(f"cuda:{device}")
    model = YOLO(str(checkpoint)).model.to(torch_device)
    if int(model.model[-1].nc) != len(SNI21_CLASSES):
        raise ValueError("Checkpoint bukan detector SNI-21")
    model.args = get_cfg(
        overrides={"epochs": 50, "box": 7.5, "cls": 0.5, "dfl": 1.5}
    )
    model.train()
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.eval()

    AuditLoss = _make_audit_loss(model, DEFAULT_ONTOLOGY_CONFIG)
    criterion = E2ELoss(
        model,
        loss_fn=lambda bound_model, tal_topk=10, tal_topk2=None: AuditLoss(
            tal_topk=tal_topk, tal_topk2=tal_topk2
        ),
    )
    schedule_weights = {"one2many": 0.45, "one2one": 0.55}
    named_parameters, parameter_groups = _parameter_groups(model)
    rows = []
    for batch_index, batch in enumerate(loader, 1):
        if batch_index > max_batches:
            break
        prepared = {
            key: (
                value.to(torch_device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in batch.items()
        }
        prepared["img"] = prepared["img"].float() / 255.0
        predictions = model(prepared["img"])
        criterion(predictions, prepared)
        o2m = criterion.one2many
        o2o = criterion.one2one
        components = (
            o2m.last_flat_classification,
            o2m.last_ontology,
            o2o.last_flat_classification,
            o2o.last_ontology,
        )
        if any(component is None for component in components):
            raise RuntimeError("Komponen gradient audit tidak tertangkap")
        flat = (
            schedule_weights["one2many"] * o2m.last_flat_classification
            + schedule_weights["one2one"] * o2o.last_flat_classification
        )
        ontology = (
            schedule_weights["one2many"] * o2m.last_ontology
            + schedule_weights["one2one"] * o2o.last_ontology
        )
        geometry = _gradient_geometry(
            flat, ontology, named_parameters, parameter_groups
        )
        row = {
            "batch": batch_index,
            "images": int(prepared["img"].shape[0]),
            "targets": int(prepared["cls"].numel()),
            "foreground_one2many": o2m.last_foreground,
            "foreground_one2one": o2o.last_foreground,
            "flat_classification_loss": float(flat.detach().cpu()),
            "ontology_loss": float(ontology.detach().cpu()),
            "groups": geometry,
        }
        rows.append(row)
        print(
            f"GRADIENT AUDIT {batch_index}/{max_batches} | "
            f"trunk={geometry['feature_extractor']['cosine']:+.4f} | "
            f"head={geometry['classification_head']['cosine']:+.4f}",
            flush=True,
        )

    if len(rows) != max_batches:
        raise RuntimeError(
            f"Loader hanya menghasilkan {len(rows)} dari {max_batches} batch"
        )
    summaries = {}
    for group_name in parameter_groups:
        summaries[group_name] = summarize_cosines(
            row["groups"][group_name]["cosine"] for row in rows
        )
        summaries[group_name]["median_flat_norm"] = float(
            median(row["groups"][group_name]["flat_norm"] for row in rows)
        )
        summaries[group_name]["median_ontology_norm"] = float(
            median(row["groups"][group_name]["ontology_norm"] for row in rows)
        )
        summaries[group_name]["shared_parameters"] = int(
            rows[0]["groups"][group_name]["shared_parameters"]
        )
    decision = route_conflict_decision(
        summaries["feature_extractor"], summaries["classification_head"]
    )
    payload = {
        "protocol": "sni21-gradient-conflict-audit-v1",
        "checkpoint": str(checkpoint),
        "dataset_root": str(layout.root),
        "split": "train",
        "training_executed": False,
        "validation_images_accessed": False,
        "test_images_accessed": False,
        "seed": seed,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "sampled_images": batch_size * max_batches,
        "image_size": 640,
        "runtime_augmentation": False,
        "branch_weights": schedule_weights,
        "ontology_config": DEFAULT_ONTOLOGY_CONFIG.to_dict(),
        "summaries": summaries,
        "decision": decision,
        "batches": rows,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train-only D0 leaf-versus-ontology gradient conflict audit"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_ontology_gradient_conflict_audit(
        args.checkpoint,
        args.data_root,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        seed=args.seed,
    )
    print(json.dumps(result["summaries"], indent=2, ensure_ascii=False))
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False))
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
