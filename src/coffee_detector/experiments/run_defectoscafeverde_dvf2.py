"""Train one selective symmetric dual-view fuser from frozen DVF1 caches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from coffee_detector.defectos_dual_view import (
    DualViewFusionConfig,
    SelectiveDualViewFuser,
    SelectiveDualViewFusionConfig,
    SymmetricDualViewFuser,
)
from coffee_detector.defectos_dual_view.model import paper_base_logits
from coffee_detector.experiments.run_defectoscafeverde_dual_view_fusion import (
    FORMAT as DVF1_FORMAT,
    _device,
    _json,
    _metrics,
    _sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/defectoscafeverde/DVF2.yaml"
PROTOCOL = "defectoscafeverde-selective-dual-view-fusion-seed42-v1"
FORMAT = "coffee_detector.defectoscafeverde.dvf2.result.v1"
REVIEW_FORMAT = "coffee_detector.defectoscafeverde.dvf1.review.v1"
METRICS = (
    "physical_pair_accuracy",
    "macro_class_accuracy",
    "bottom3_class_accuracy",
    "worst_class_accuracy",
)


def _config() -> tuple[dict, SelectiveDualViewFusionConfig]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if payload.get("code") != "DVF2" or payload.get("protocol") != PROTOCOL:
        raise RuntimeError("Konfigurasi DVF2 berubah")
    return payload, SelectiveDualViewFusionConfig.from_mapping(payload["fuser"])


def run_static_audit(output: str | Path) -> dict:
    payload, config = _config()
    torch.manual_seed(42)
    logits = torch.randn(8, 2, config.classes, requires_grad=True)
    labels = torch.arange(8) % config.classes
    model = SelectiveDualViewFuser(config)
    base = paper_base_logits(logits)
    initial, details = model(logits, return_details=True)
    swapped = model(logits.flip(1))
    gate_target = (base.argmax(dim=1) != labels).float()
    loss = F.cross_entropy(initial, labels) + F.binary_cross_entropy_with_logits(
        details["gate_logit"], gate_target
    )
    loss.backward()
    with torch.no_grad():
        model.expert[-1].bias.fill_(0.25)
        model.gate[-1].weight.zero_()
        model.gate[-1].bias.fill_(-10.0)
        model.eval()
        fallback = model(logits.detach())
        model.gate[-1].bias.fill_(10.0)
        active = model(logits.detach())
    parameters = sum(parameter.numel() for parameter in model.parameters())
    gate_gradients = [parameter.grad for parameter in model.gate.parameters()]
    gates = {
        "config_exact": payload["code"] == "DVF2" and payload["protocol"] == PROTOCOL,
        "parameter_count_under_15000": 0 < parameters < 15_000,
        "initial_output_exactly_paper_rule": torch.equal(initial, base),
        "initial_swap_invariant": torch.equal(initial, swapped),
        "gate_negative_fallback_exact": torch.equal(fallback, base.detach()),
        "gate_positive_expert_changes_output": not torch.equal(active, base.detach()),
        "expert_final_gradient_finite_nonzero": bool(
            model.expert[-1].weight.grad is not None
            and torch.isfinite(model.expert[-1].weight.grad).all()
            and model.expert[-1].weight.grad.abs().sum() > 0
        ),
        "gate_gradients_finite_nonzero": all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gate_gradients
        )
        and any(gradient.abs().sum() > 0 for gradient in gate_gradients),
        "detector_not_loaded_or_trained": True,
        "test_not_accessed": True,
    }
    result = {
        "format": "coffee_detector.defectoscafeverde.dvf2.static_audit.v1",
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


def _validate_dvf1(result_path: Path, review_path: Path) -> tuple[dict, dict, Path, Path, Path]:
    result = _json(result_path, "DVF1 result")
    review = _json(review_path, "DVF1 review")
    if (
        result.get("format") != DVF1_FORMAT
        or result.get("decision") != "PASS"
        or result.get("test_images_accessed") is not False
    ):
        raise RuntimeError("DVF1 result tidak valid")
    if (
        review.get("format") != REVIEW_FORMAT
        or review.get("decision") != "AUTHORIZE_DVF1_FUSER_MULTISEED"
        or review.get("training_executed") is not False
        or review.get("test_images_accessed") is not False
    ):
        raise RuntimeError("DVF1 review tidak valid")
    root = result_path.parent
    checkpoint = root / "DVF1_last.pt"
    train_cache = root / "train_pair_cache.pt"
    val_cache = root / "val_pair_cache.pt"
    for path in (checkpoint, train_cache, val_cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(result_path) != review["artifacts"]["result_sha256"]:
        raise RuntimeError("SHA DVF1 result tidak cocok dengan review")
    if _sha256(checkpoint) != review["artifacts"]["checkpoint_sha256"]:
        raise RuntimeError("SHA DVF1 checkpoint tidak cocok dengan review")
    if _sha256(val_cache) != review["artifacts"]["validation_cache_sha256"]:
        raise RuntimeError("SHA validation cache tidak cocok dengan review")
    return result, review, checkpoint, train_cache, val_cache


def _train_fuser(
    cache: dict,
    config: SelectiveDualViewFusionConfig,
    train_config: Mapping,
    output: Path,
    contract: dict,
    device: torch.device,
) -> tuple[SelectiveDualViewFuser, list[dict], bool]:
    seed = int(train_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = SelectiveDualViewFuser(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    epochs = int(train_config["epochs"])
    start, history, generator_state = 0, [], None
    if output.is_file():
        state = torch.load(output, map_location=device, weights_only=False)
        if state.get("contract") != contract:
            raise RuntimeError("Checkpoint DVF2 stale")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, history = int(state["epoch"]), list(state["history"])
        generator_state = state.get("generator_state")
        if start >= epochs:
            print(f"REUSE COMPLETE DVF2 FUSER: {output}", flush=True)
            return model.eval(), history, False
    labels = cache["labels"].long()
    counts = torch.bincount(labels, minlength=config.classes).float().clamp_min(1)
    class_weight = (counts.rsqrt() / counts.rsqrt().mean()).to(device)
    base_wrong = paper_base_logits(cache["view_logits"].float()).argmax(dim=1) != labels
    positives = int(base_wrong.sum())
    negatives = int((~base_wrong).sum())
    if positives == 0 or negatives == 0:
        raise RuntimeError("Gate target train tidak memiliki dua kelas")
    gate_pos_weight = torch.tensor(
        min(10.0, negatives / positives), dtype=torch.float32, device=device
    )
    dataset = TensorDataset(cache["view_logits"].float(), labels)
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
        model.train()
        totals = {key: 0.0 for key in ("loss", "ce", "gate", "rescue", "preserve")}
        seen = 0
        for pair_logits, target in loader:
            pair_logits, target = pair_logits.to(device), target.to(device)
            fused, details = model(pair_logits, return_details=True)
            wrong = details["base"].detach().argmax(dim=1) != target
            ce = F.cross_entropy(
                fused,
                target,
                weight=class_weight,
                label_smoothing=float(train_config["label_smoothing"]),
            )
            gate_loss = F.binary_cross_entropy_with_logits(
                details["gate_logit"], wrong.float(), pos_weight=gate_pos_weight
            )
            rescue = (
                F.cross_entropy(
                    fused[wrong],
                    target[wrong],
                    weight=class_weight,
                    label_smoothing=float(train_config["label_smoothing"]),
                )
                if bool(wrong.any())
                else fused.sum() * 0.0
            )
            correct = ~wrong
            preserve = (
                F.kl_div(
                    F.log_softmax(fused[correct], dim=1),
                    F.softmax(details["base"].detach()[correct], dim=1),
                    reduction="batchmean",
                )
                if bool(correct.any())
                else fused.sum() * 0.0
            )
            loss = (
                ce
                + float(train_config["gate_loss_weight"]) * gate_loss
                + float(train_config["rescue_loss_weight"]) * rescue
                + float(train_config["preservation_weight"]) * preserve
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(target)
            seen += count
            for key, value in (
                ("loss", loss),
                ("ce", ce),
                ("gate", gate_loss),
                ("rescue", rescue),
                ("preserve", preserve),
            ):
                totals[key] += float(value.detach()) * count
        row = {"epoch": epoch + 1, **{key: value / seen for key, value in totals.items()}}
        history.append(row)
        torch.save(
            {
                "format": "coffee_detector.defectoscafeverde.dvf2.checkpoint.v1",
                "contract": contract,
                "config": config.to_dict(),
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "history": history,
                "train_gate_targets": {"positive": positives, "negative": negatives},
            },
            output,
        )
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"DVF2 FUSER {epoch + 1}/{epochs} loss={row['loss']:.6f} "
                f"gate={row['gate']:.6f}",
                flush=True,
            )
    return model.eval(), history, True


def _correctness(fused_logits: torch.Tensor, cache: dict, device: torch.device) -> torch.Tensor:
    view_logits = cache["view_logits"].to(device)
    scores = view_logits.sigmoid()
    labels = cache["labels"].to(device)
    ious = cache["ious"].to(device)
    rows = torch.arange(len(labels), device=device)
    fused_class = fused_logits.argmax(dim=1)
    class_support = scores.gather(
        2, fused_class[:, None, None].expand(-1, 2, 1)
    ).squeeze(2)
    fused_view = class_support.argmax(dim=1)
    return (fused_class == labels) & (ious[rows, fused_view] >= 0.5)


def evaluate(
    dvf1: SymmetricDualViewFuser,
    dvf2: SelectiveDualViewFuser,
    cache: dict,
    device: torch.device,
) -> tuple[dict, dict]:
    logits = cache["view_logits"].to(device)
    labels = cache["labels"].to(device)
    ious = cache["ious"].to(device)
    names = {int(key): str(value) for key, value in cache["names"].items()}
    scores = logits.sigmoid()
    view_confidence, view_class = scores.max(dim=2)
    paper_view = view_confidence.argmax(dim=1)
    rows = torch.arange(len(labels), device=device)
    paper_class = view_class[rows, paper_view]
    paper_correct = (paper_class == labels) & (ious[rows, paper_view] >= 0.5)
    side_correct = (view_class == labels[:, None]) & (ious >= 0.5)
    with torch.inference_mode():
        dvf1_logits = dvf1.eval()(logits)
        dvf2_logits, details = dvf2.eval()(logits, return_details=True)
    dvf1_correct = _correctness(dvf1_logits, cache, device)
    dvf2_correct = _correctness(dvf2_logits, cache, device)
    active = details["effective_gate"].bool()
    values = {
        "PAPER_MAX_CONFIDENCE": _metrics(paper_correct, labels, names),
        "DVF1": _metrics(dvf1_correct, labels, names),
        "DVF2": _metrics(dvf2_correct, labels, names),
        "PAIR_ORACLE": _metrics(side_correct.any(dim=1), labels, names),
    }
    transitions = {
        "rescue": int(((~dvf1_correct) & dvf2_correct).sum()),
        "regression": int((dvf1_correct & (~dvf2_correct)).sum()),
        "both_correct": int((dvf1_correct & dvf2_correct).sum()),
        "both_wrong": int(((~dvf1_correct) & (~dvf2_correct)).sum()),
    }
    gate = {
        "active_pairs": int(active.sum()),
        "active_fraction": float(active.float().mean()),
        "active_on_paper_correct": int((active & paper_correct).sum()),
        "paper_correct_pairs": int(paper_correct.sum()),
        "active_on_paper_wrong": int((active & (~paper_correct)).sum()),
        "paper_wrong_pairs": int((~paper_correct).sum()),
        "mean_probability": float(details["gate_probability"].mean()),
        "transitions_vs_dvf1": transitions,
    }
    return values, gate


def _decision(values: Mapping[str, Mapping], config: Mapping) -> tuple[dict, str, str]:
    dvf1, dvf2 = values["DVF1"], values["DVF2"]
    deltas = {metric: dvf2[metric] - dvf1[metric] for metric in METRICS}
    common = (
        deltas["macro_class_accuracy"] >= float(config["macro_floor_vs_dvf1"])
        and deltas["worst_class_accuracy"]
        >= -float(config["worst_drop_limit_vs_dvf1"])
    )
    overall_route = (
        deltas["physical_pair_accuracy"]
        >= float(config["overall_gain_over_dvf1"])
        and deltas["bottom3_class_accuracy"]
        >= float(config["bottom3_floor_vs_dvf1"])
        and common
    )
    tail_route = (
        deltas["physical_pair_accuracy"] >= 0.0
        and deltas["bottom3_class_accuracy"]
        >= float(config["tail_gain_over_dvf1"])
        and common
    )
    passed = overall_route or tail_route
    comparison = {
        "deltas_vs_dvf1": deltas,
        "overall_route": overall_route,
        "lower_tail_route": tail_route,
        "criteria": {
            "macro_not_lower_than_dvf1": deltas["macro_class_accuracy"] >= 0.0,
            "worst_drop_no_more_than_2_points": deltas["worst_class_accuracy"] >= -0.02,
        },
    }
    return (
        comparison,
        "PASS" if passed else "FAIL",
        "REVIEW_DVF2_BEFORE_CONFIRMATION" if passed else "RETAIN_DVF1",
    )


def run_dvf2(
    dvf1_result: str | Path,
    dvf1_review: str | Path,
    output_root: str | Path,
    *,
    device: str = "cpu",
    authorize_training: bool = False,
) -> dict:
    if not authorize_training:
        raise PermissionError("DVF2 memerlukan --authorize-training")
    result_path = Path(dvf1_result).expanduser().resolve()
    review_path = Path(dvf1_review).expanduser().resolve()
    dvf1_result_payload, _, dvf1_checkpoint, train_path, val_path = _validate_dvf1(
        result_path, review_path
    )
    payload, config = _config()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "DVF2_seed42_result.json"
    contract = {
        "protocol": PROTOCOL,
        "config_sha256": _sha256(CONFIG_PATH),
        "dvf1_result_sha256": _sha256(result_path),
        "dvf1_review_sha256": _sha256(review_path),
        "dvf1_checkpoint_sha256": _sha256(dvf1_checkpoint),
        "train_cache_sha256": _sha256(train_path),
        "validation_cache_sha256": _sha256(val_path),
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    if destination.is_file():
        cached = _json(destination, "Cached DVF2 result")
        if cached.get("format") != FORMAT or cached.get("run_contract") != contract:
            raise RuntimeError("Cached DVF2 result stale")
        cached["training_executed_this_call"] = False
        print(f"REUSE COMPLETE DVF2 RESULT: {destination}", flush=True)
        return cached
    static = run_static_audit(output_root / "static_audit.json")
    if static["decision"] != "PASS":
        raise RuntimeError(f"Static audit DVF2 gagal: {static['gates']}")
    torch_device = _device(device)
    train_cache = torch.load(train_path, map_location="cpu", weights_only=False)
    val_cache = torch.load(val_path, map_location="cpu", weights_only=False)
    if train_cache.get("split") != "train" or val_cache.get("split") != "val":
        raise RuntimeError("Split cache DVF1 tidak valid")
    if set(train_cache["group_ids"]) & set(val_cache["group_ids"]):
        raise RuntimeError("Train/validation physical groups tumpang tindih")
    train_sha_before, val_sha_before = _sha256(train_path), _sha256(val_path)
    checkpoint = torch.load(dvf1_checkpoint, map_location="cpu", weights_only=False)
    dvf1 = SymmetricDualViewFuser(
        DualViewFusionConfig.from_mapping(checkpoint["config"])
    )
    dvf1.load_state_dict(checkpoint["model"])
    dvf1.to(torch_device).eval()
    fuser_path = output_root / "DVF2_last.pt"
    fuser, history, trained = _train_fuser(
        train_cache,
        config,
        payload["train"],
        fuser_path,
        contract,
        torch_device,
    )
    values, gate = evaluate(dvf1, fuser, val_cache, torch_device)
    endpoint_calibration = {
        endpoint: {
            metric: values[endpoint][metric]
            - dvf1_result_payload["values"][endpoint][metric]
            for metric in METRICS
        }
        for endpoint in ("PAPER_MAX_CONFIDENCE", "DVF1", "PAIR_ORACLE")
    }
    gates = {
        "static_audit_pass": static["decision"] == "PASS",
        "dvf1_endpoints_reproduced_exactly": all(
            value == 0.0
            for endpoint in endpoint_calibration.values()
            for value in endpoint.values()
        ),
        "train_validation_groups_disjoint": not (
            set(train_cache["group_ids"]) & set(val_cache["group_ids"])
        ),
        "train_cache_unchanged": _sha256(train_path) == train_sha_before,
        "validation_cache_unchanged": _sha256(val_path) == val_sha_before,
        "all_12_validation_classes_present": all(
            not row["classes_without_ground_truth"] for row in values.values()
        ),
        "detector_not_loaded_or_trained": True,
        "test_not_opened": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"DVF2 runtime gate gagal: {gates}")
    comparison, decision, next_action = _decision(values, payload["decision"])
    result = {
        "format": FORMAT,
        "protocol": PROTOCOL,
        "seed": 42,
        "values": values,
        "comparison": comparison,
        "gate_diagnostics": gate,
        "endpoint_calibration": endpoint_calibration,
        "gates": gates,
        "decision": decision,
        "next": next_action,
        "fuser_parameters": sum(parameter.numel() for parameter in fuser.parameters()),
        "history": history,
        "artifacts": {
            "fuser_checkpoint": str(fuser_path),
            "fuser_checkpoint_sha256": _sha256(fuser_path),
            "static_audit": str(output_root / "static_audit.json"),
        },
        "run_contract": contract,
        "scientific_status": "exploratory_after_reused_validation_review",
        "detector_training_executed": False,
        "fuser_training_executed": True,
        "training_executed_this_call": trained,
        "test_images_accessed": False,
    }
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="DVF2 selective dual-view fusion")
    parser.add_argument("--dvf1-result")
    parser.add_argument("--dvf1-review")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    if args.static_only:
        result = run_static_audit(Path(args.output_root) / "static_audit.json")
        print(json.dumps(result, indent=2))
        return
    if args.dvf1_result is None or args.dvf1_review is None:
        parser.error("--dvf1-result dan --dvf1-review wajib")
    run_dvf2(
        args.dvf1_result,
        args.dvf1_review,
        args.output_root,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
