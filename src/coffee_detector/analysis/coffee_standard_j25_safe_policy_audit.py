"""No-training audit to prioritize the next J25 safe-policy ablation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from coffee_detector.af2_luminance import (
    identity_repeat_factor_weights,
    j25_source_identity,
)
from coffee_detector.analysis.coffee_fg_diagnostics import diagnose_checkpoint
from coffee_detector.analysis.coffee_standard_j25_black_broken_audit import (
    TARGET_CLASS,
    _target_view,
    attribute_target,
)
from coffee_detector.dataset import IMAGE_SUFFIXES, discover_layout, parse_label
from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import (
    _json,
    validate_j25_development,
)
from coffee_detector.experiments.run_coffee_standard_j25_af2_luminance_safe import (
    PROTOCOL as SAFE_PROTOCOL,
)
from coffee_detector.experiments.run_coffee_standard_j25_af2_luminance_strength_sweep import (
    PROTOCOL as STRENGTH_PROTOCOL,
)
from coffee_detector.experiments.run_coffee_standard_j25_safe_d0 import (
    DIRECT_PROTOCOL,
    PROTOCOL as CONTROL_PROTOCOL,
)
from coffee_detector.experiments.run_faruq_v3_af2_direct import _sha256


PROTOCOL = "coffee-standard-j25-safe-policy-root-cause-v1"


def target_training_support(root: str | Path, maximum_repeat: float = 4.0) -> dict:
    layout = discover_layout(root)
    target_ids = [index for index, name in layout.names.items() if name == TARGET_CLASS]
    if len(target_ids) != 1:
        raise RuntimeError(f"Target class ambigu/tidak ada: {target_ids}")
    target_id = target_ids[0]
    image_root, label_root = layout.splits["train"]
    images = sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    class_sets: list[set[int]] = []
    target_indexes: list[int] = []
    target_instances = 0
    target_identities: Counter[str] = Counter()
    cooccurrence: Counter[int] = Counter()
    for index, image in enumerate(images):
        annotations = parse_label(
            (label_root / image.relative_to(image_root)).with_suffix(".txt"),
            set(layout.names),
        )
        classes = {item.class_id for item in annotations}
        class_sets.append(classes)
        count = sum(item.class_id == target_id for item in annotations)
        if count:
            target_indexes.append(index)
            target_instances += count
            target_identities[j25_source_identity(image)] += 1
            cooccurrence.update(classes - {target_id})
    weights, sampler = identity_repeat_factor_weights(
        images,
        class_sets,
        class_count=len(layout.names),
        maximum_repeat=maximum_repeat,
    )
    target_weight = float(weights[target_indexes].sum()) if target_indexes else 0.0
    natural_share = len(target_indexes) / max(len(images), 1)
    weighted_share = target_weight / float(weights.sum())
    derivatives = list(target_identities.values())
    return {
        "class_id": target_id,
        "class_name": TARGET_CLASS,
        "train_images": len(target_indexes),
        "train_instances": target_instances,
        "train_source_identities": len(target_identities),
        "derivatives_per_identity_min": min(derivatives) if derivatives else 0,
        "derivatives_per_identity_max": max(derivatives) if derivatives else 0,
        "natural_image_share": natural_share,
        "sampler_expected_draw_share": weighted_share,
        "sampler_exposure_multiplier": weighted_share / max(natural_share, 1e-12),
        "class_identity_count": sampler["class_identity_counts"][str(target_id)],
        "class_repeat_factor": sampler["class_repeat_factors"][str(target_id)],
        "cooccurring_target_image_counts": {
            layout.names[index]: int(count)
            for index, count in cooccurrence.most_common()
        },
    }


def prioritize_ablation(models: dict, support: dict) -> dict:
    d0 = models["D0DIRECT"]["raw_top500"]
    control = models["SAFED0"]["raw_top500"]
    degraded = control["correct_class"] < d0["correct_class"]
    sampler_shift = float(support["sampler_exposure_multiplier"]) > 1.05
    if degraded and sampler_shift:
        next_arm = "SAFE_AUGMENTATION_ONLY_NO_SAMPLER"
        reason = "remove_sampler_first_to_test_distribution_shift"
    elif degraded:
        next_arm = "IDENTITY_SAMPLER_ONLY_STANDARD_AUGMENTATION"
        reason = "sampler_barely_changes_target_exposure_test_augmentation_first"
    else:
        next_arm = "NO_NEW_TRAINING_REVIEW_LABELS_AND_RANKING"
        reason = "safe_policy_did_not_reduce_raw_correct_target_count"
    return {
        "raw_correct_degraded_vs_d0": degraded,
        "sampler_target_exposure_shift_over_5_percent": sampler_shift,
        "priority_arm": next_arm,
        "reason": reason,
        "boundary": "diagnostic prioritization, not causal proof",
    }


def run_safe_policy_audit(
    data_root: str | Path,
    development_contract: str | Path,
    provenance_summary: str | Path,
    d0_checkpoint: str | Path,
    d0_result: str | Path,
    control_checkpoint: str | Path,
    control_result: str | Path,
    safe_checkpoint: str | Path,
    safe_result: str | Path,
    strength_summary: str | Path,
    sampler_audit: str | Path,
    output: str | Path,
    *,
    device: str = "0",
    authorize_diagnostic: bool = False,
) -> dict:
    if not authorize_diagnostic:
        raise RuntimeError("Audit harus diotorisasi eksplisit")
    root = Path(data_root).expanduser().resolve()
    dataset = validate_j25_development(root, development_contract, provenance_summary)
    if not dataset["data_format"].endswith("train_siblings.v2"):
        raise RuntimeError("Audit hanya untuk J25 train-siblings v2")
    checkpoints = {
        "D0DIRECT": Path(d0_checkpoint).expanduser().resolve(),
        "SAFED0": Path(control_checkpoint).expanduser().resolve(),
        "AF2LUMSAFE_lambda0": Path(safe_checkpoint).expanduser().resolve(),
    }
    results = {
        "D0DIRECT": _json(d0_result, "D0 result"),
        "SAFED0": _json(control_result, "SAFED0 result"),
        "AF2LUMSAFE_lambda0": _json(safe_result, "AF2LUMSAFE result"),
    }
    protocols = {
        "D0DIRECT": DIRECT_PROTOCOL,
        "SAFED0": CONTROL_PROTOCOL,
        "AF2LUMSAFE_lambda0": SAFE_PROTOCOL,
    }
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result = results[name]
        if result.get("protocol") != protocols[name] or result.get("test_images_accessed") is not False:
            raise RuntimeError(f"Result contract salah: {name}")
        if result.get("checkpoint_sha256") != _sha256(checkpoint):
            raise RuntimeError(f"Checkpoint SHA berbeda: {name}")
    strength = _json(strength_summary, "Strength summary")
    if (
        strength.get("protocol") != STRENGTH_PROTOCOL
        or strength.get("test_opened") is not False
        or strength.get("checkpoint_sha256") != _sha256(checkpoints["AF2LUMSAFE_lambda0"])
    ):
        raise RuntimeError("Strength summary contract salah")
    sampler = _json(sampler_audit, "SAFED0 sampler audit")
    if sampler.get("protocol") != CONTROL_PROTOCOL or sampler.get("test_images_accessed") is not False:
        raise RuntimeError("Sampler audit contract salah")

    support = target_training_support(root, maximum_repeat=float(sampler["maximum_repeat"]))
    if (
        support["class_identity_count"]
        != sampler["class_identity_counts"][str(support["class_id"])]
        or support["class_repeat_factor"]
        != sampler["class_repeat_factors"][str(support["class_id"])]
    ):
        raise RuntimeError("Target support berbeda dari sampler audit")

    diagnostics, views = {}, {}
    for name, checkpoint in checkpoints.items():
        diagnostic = diagnose_checkpoint(
            checkpoint,
            root,
            split="val",
            image_size=640,
            candidate_counts=(500,),
            iou_threshold=0.5,
            confidence_threshold=0.001,
            nms_iou=0.7,
            max_det=500,
            device=device,
            inference_strength=0.0 if name == "AF2LUMSAFE_lambda0" else None,
        )
        diagnostics[name] = diagnostic
        views[name] = _target_view(diagnostic)
        print(name, json.dumps(views[name], ensure_ascii=False), flush=True)

    payload = {
        "format": "coffee_detector.j25.safe_policy_root_cause.v1",
        "protocol": PROTOCOL,
        "target_training_support": support,
        "models": views,
        "attribution": {name: attribute_target(view) for name, view in views.items()},
        "next_ablation_priority": prioritize_ablation(views, support),
        "diagnostic_settings": {
            "raw_candidates": 500,
            "iou_threshold": 0.5,
            "final_confidence": 0.001,
            "safe_inference_strength": 0.0,
        },
        "training_executed": False,
        "test_opened": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    destination.with_name(destination.stem + "_full.json").write_text(
        json.dumps({"summary": payload, "full_diagnostics": diagnostics}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--development-contract", required=True)
    parser.add_argument("--provenance-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--d0-result", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--control-result", required=True)
    parser.add_argument("--safe-checkpoint", required=True)
    parser.add_argument("--safe-result", required=True)
    parser.add_argument("--strength-summary", required=True)
    parser.add_argument("--sampler-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-diagnostic", action="store_true")
    args = parser.parse_args()
    run_safe_policy_audit(
        args.data_root,
        args.development_contract,
        args.provenance_summary,
        args.d0_checkpoint,
        args.d0_result,
        args.control_checkpoint,
        args.control_result,
        args.safe_checkpoint,
        args.safe_result,
        args.strength_summary,
        args.sampler_audit,
        args.output,
        device=args.device,
        authorize_diagnostic=args.authorize_diagnostic,
    )


if __name__ == "__main__":
    main()
