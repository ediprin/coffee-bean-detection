"""Validation-only paired-side audit for DefectosCafeVerde.

The source paper captures two faces of one physical bean and combines the two
independent detector decisions with a maximum-confidence rule.  The grouped
rebuild retained those physical pairs.  This audit measures whether the
completed native detector contains complementary evidence across the two
faces before any learned cross-view architecture is trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torchvision.ops import box_iou

from coffee_detector.analysis.coffee_fg_diagnostics import (
    _letterbox_sample,
    _raw_branches,
    _split_samples,
    _unwrap_head,
)


FORMAT = "coffee_detector.defectoscafeverde.dual_view_audit.v1"
REFERENCE_PROTOCOL = "defectoscafeverde-grouped-dcwcf-direct-seed42-v1"
SOURCE_NUMBER = re.compile(r"^(?P<prefix>[A-Za-z]+)(?P<number>\d+)")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path, label: str) -> Any:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label}: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _source_order(source_name: str) -> tuple[str, int, str]:
    match = SOURCE_NUMBER.match(Path(source_name).name)
    if not match:
        return (source_name.casefold(), 0, source_name.casefold())
    return (
        match.group("prefix").casefold(),
        int(match.group("number")),
        source_name.casefold(),
    )


def _validate_reference(result_path: str | Path, checkpoint_path: str | Path) -> dict:
    result = _load_json(result_path, "D0DIRECT result")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"D0DIRECT checkpoint: {checkpoint}")
    contract = result.get("run_contract", {})
    gates = {
        "reference_arm_seed_exact": result.get("arm") == "D0DIRECT"
        and int(result.get("seed", -1)) == 42,
        "reference_protocol_exact": result.get("protocol") == REFERENCE_PROTOCOL,
        "reference_is_completed_validation": result.get("evaluation_split") == "val"
        and int(result.get("completed_epochs", -1)) == 50,
        "checkpoint_sha256_exact": _sha256(checkpoint)
        == result.get("checkpoint_sha256"),
        "dataset_audit_sha_present": bool(contract.get("dataset_audit_sha256")),
        "reference_test_not_opened": result.get("test_images_accessed") is False,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Kontrak D0DIRECT gagal: {gates}")
    return {"result": result, "checkpoint": checkpoint, "gates": gates}


def _manifest_pairs(
    manifest_path: str | Path,
    samples: list[tuple[Path, tuple]],
) -> tuple[dict[str, list[tuple[Path, tuple, dict]]], dict[str, Any]]:
    payload = _load_json(manifest_path, "Grouped manifest")
    if not isinstance(payload, list):
        raise TypeError("Grouped manifest harus berupa list")
    validation_rows = [row for row in payload if row.get("output_split") == "val"]
    by_name = {Path(str(row["image"])).name: row for row in validation_rows}
    if len(by_name) != len(validation_rows):
        raise RuntimeError("Nama citra validation pada manifest tidak unik")
    grouped: dict[str, list[tuple[Path, tuple, dict]]] = defaultdict(list)
    for image_path, annotations in samples:
        row = by_name.get(image_path.name)
        if row is None:
            raise RuntimeError(f"Citra validation tidak ditemukan di manifest: {image_path.name}")
        grouped[str(row["physical_group_id"])].append((image_path, annotations, row))
    if sum(len(rows) for rows in grouped.values()) != len(samples):
        raise RuntimeError("Pemetaan manifest validation tidak lengkap")
    sizes = Counter(len(rows) for rows in grouped.values())
    return dict(grouped), {
        "validation_manifest_rows": len(validation_rows),
        "validation_groups": len(grouped),
        "group_size_histogram": {str(key): value for key, value in sorted(sizes.items())},
    }


def _eligible_pairs(
    grouped: Mapping[str, list[tuple[Path, tuple, dict]]]
) -> tuple[list[tuple[str, list[tuple[Path, tuple, dict]], int]], Counter[str]]:
    eligible = []
    rejected: Counter[str] = Counter()
    for group_id, rows in grouped.items():
        if len(rows) != 2:
            rejected["not_exactly_two_views"] += 1
            continue
        if any(len(annotations) != 1 for _, annotations, _ in rows):
            rejected["not_one_object_per_view"] += 1
            continue
        classes = [int(annotations[0].class_id) for _, annotations, _ in rows]
        if classes[0] != classes[1]:
            rejected["paired_labels_disagree"] += 1
            continue
        ordered = sorted(rows, key=lambda item: _source_order(str(item[2]["source_name"])))
        eligible.append((group_id, ordered, classes[0]))
    return eligible, rejected


def _top_decision(
    network: torch.nn.Module,
    image_path: Path,
    annotations: tuple,
    *,
    image_size: int,
    confidence: float,
    max_det: int,
    iou_threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    image, target_boxes, target_labels, _ = _letterbox_sample(
        image_path, annotations, image_size, device
    )
    final, _, _ = _raw_branches(network, image, max_det)
    kept = final[final[:, 4] >= confidence]
    if not len(kept):
        return {
            "predicted_class_id": None,
            "confidence": 0.0,
            "iou": 0.0,
            "localized": False,
            "correct": False,
        }
    index = int(kept[:, 4].argmax())
    candidate = kept[index]
    iou = float(box_iou(candidate[None, :4], target_boxes).max())
    predicted = int(candidate[5])
    expected = int(target_labels[0])
    return {
        "predicted_class_id": predicted,
        "confidence": float(candidate[4]),
        "iou": iou,
        "localized": iou >= iou_threshold,
        "correct": iou >= iou_threshold and predicted == expected,
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def summarize_pairs(records: list[dict], names: Mapping[int, str]) -> dict:
    total = len(records)
    side_a = sum(row["views"][0]["correct"] for row in records)
    side_b = sum(row["views"][1]["correct"] for row in records)
    max_conf = sum(row["paper_max_confidence_correct"] for row in records)
    oracle = sum(row["pair_oracle_correct"] for row in records)
    both = sum(all(view["correct"] for view in row["views"]) for row in records)
    exactly_one = sum(sum(view["correct"] for view in row["views"]) == 1 for row in records)
    neither = total - both - exactly_one
    best_single = max(side_a, side_b)
    classwise = {}
    complementary_classes = 0
    for class_id, class_name in names.items():
        rows = [row for row in records if row["class_id"] == class_id]
        if not rows:
            continue
        a = sum(row["views"][0]["correct"] for row in rows)
        b = sum(row["views"][1]["correct"] for row in rows)
        paper = sum(row["paper_max_confidence_correct"] for row in rows)
        upper = sum(row["pair_oracle_correct"] for row in rows)
        one = sum(sum(view["correct"] for view in row["views"]) == 1 for row in rows)
        if one:
            complementary_classes += 1
        classwise[class_name] = {
            "groups": len(rows),
            "side_a_accuracy": _rate(a, len(rows)),
            "side_b_accuracy": _rate(b, len(rows)),
            "paper_max_confidence_accuracy": _rate(paper, len(rows)),
            "pair_oracle_accuracy": _rate(upper, len(rows)),
            "exactly_one_side_correct": one,
        }
    return {
        "eligible_groups": total,
        "side_a_accuracy": _rate(side_a, total),
        "side_b_accuracy": _rate(side_b, total),
        "best_single_side_accuracy": _rate(best_single, total),
        "paper_max_confidence_accuracy": _rate(max_conf, total),
        "pair_oracle_accuracy": _rate(oracle, total),
        "paper_gain_over_best_single_side": _rate(max_conf - best_single, total),
        "pair_oracle_gain_over_best_single_side": _rate(oracle - best_single, total),
        "learned_fusion_headroom_over_paper_rule": _rate(oracle - max_conf, total),
        "both_sides_correct": both,
        "exactly_one_side_correct": exactly_one,
        "neither_side_correct": neither,
        "complementary_classes": complementary_classes,
        "classwise": classwise,
    }


def run_dual_view_audit(
    data_root: str | Path,
    grouped_manifest: str | Path,
    grouped_audit: str | Path,
    d0_result: str | Path,
    d0_checkpoint: str | Path,
    output: str | Path,
    *,
    records_output: str | Path | None = None,
    device: str = "cpu",
    image_size: int = 640,
    confidence: float = 0.001,
    iou_threshold: float = 0.5,
    max_det: int = 500,
) -> dict:
    root = Path(data_root).expanduser().resolve()
    if (root / "test").exists():
        raise RuntimeError("TEST TEREXPOSE - audit dihentikan")
    audit = _load_json(grouped_audit, "Grouped audit")
    if audit.get("decision") != "PASS_GROUPED_DATASET_GATE" or not all(
        audit.get("gates", {}).values()
    ):
        raise RuntimeError("Grouped dataset audit bukan PASS yang lengkap")
    reference = _validate_reference(d0_result, d0_checkpoint)
    expected_audit_sha = reference["result"]["run_contract"]["dataset_audit_sha256"]
    if _sha256(grouped_audit) != expected_audit_sha:
        raise RuntimeError("Grouped audit SHA berbeda dari kontrak D0DIRECT")
    layout, samples = _split_samples(root, "val")
    names = {int(index): str(name) for index, name in layout.names.items()}
    if len(names) != 12:
        raise RuntimeError(f"Ontologi harus 12 kelas, ditemukan {len(names)}")
    grouped, manifest_stats = _manifest_pairs(grouped_manifest, samples)
    eligible, rejected = _eligible_pairs(grouped)
    if not eligible:
        raise RuntimeError("Tidak ada pasangan dua-sisi satu-objek yang eligible")
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")
    from ultralytics import YOLO

    network = YOLO(str(reference["checkpoint"])).model.to(torch_device).eval()
    if int(_unwrap_head(network).nc) != len(names):
        raise RuntimeError("Checkpoint D0DIRECT tidak memiliki 12 kelas")
    records = []
    with torch.inference_mode():
        for index, (group_id, rows, class_id) in enumerate(eligible, 1):
            views = []
            for image_path, annotations, row in rows:
                outcome = _top_decision(
                    network,
                    image_path,
                    annotations,
                    image_size=image_size,
                    confidence=confidence,
                    max_det=max_det,
                    iou_threshold=iou_threshold,
                    device=torch_device,
                )
                views.append(
                    {
                        "image": str(image_path),
                        "source_name": row["source_name"],
                        **outcome,
                    }
                )
            selected = max(views, key=lambda row: row["confidence"])
            records.append(
                {
                    "physical_group_id": group_id,
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "views": views,
                    "paper_selected_source": selected["source_name"],
                    "paper_max_confidence_correct": bool(selected["correct"]),
                    "pair_oracle_correct": any(view["correct"] for view in views),
                }
            )
            if index % 64 == 0 or index == len(eligible):
                print(f"DUAL VIEW AUDIT {index}/{len(eligible)}", flush=True)
    summary = summarize_pairs(records, names)
    eligible_fraction = summary["eligible_groups"] / max(manifest_stats["validation_groups"], 1)
    class_ap = reference["result"].get("map50_95_by_class", {})
    if set(class_ap) != set(names.values()):
        raise RuntimeError("Peta AP D0DIRECT tidak cocok dengan ontologi 12 kelas")
    worst_class = min(class_ap, key=class_ap.get)
    worst_complementarity = summary["classwise"].get(worst_class, {}).get(
        "exactly_one_side_correct", 0
    )
    criteria = {
        "eligible_pairs_cover_at_least_75_percent_of_validation_groups": eligible_fraction >= 0.75,
        "pair_oracle_gain_over_best_side_at_least_1_point": summary[
            "pair_oracle_gain_over_best_single_side"
        ]
        >= 0.01,
        "learned_fusion_headroom_over_paper_rule_at_least_0_5_point": summary[
            "learned_fusion_headroom_over_paper_rule"
        ]
        >= 0.005,
        "complementarity_spans_at_least_3_classes": summary["complementary_classes"] >= 3,
        "native_worst_class_has_cross_side_complementarity": worst_complementarity > 0,
        "training_not_executed": True,
        "test_not_accessed": True,
    }
    scientific = all(
        criteria[key]
        for key in (
            "eligible_pairs_cover_at_least_75_percent_of_validation_groups",
            "pair_oracle_gain_over_best_side_at_least_1_point",
            "learned_fusion_headroom_over_paper_rule_at_least_0_5_point",
            "complementarity_spans_at_least_3_classes",
        )
    )
    decision = (
        "AUTHORIZE_ONE_LEARNED_DUAL_VIEW_ARCHITECTURE_SCREEN"
        if scientific
        else "STOP_LEARNED_DUAL_VIEW_FUSION"
    )
    payload = {
        "format": FORMAT,
        "scope": "validation_only_physical_pair_complementarity_audit",
        "settings": {
            "image_size": image_size,
            "confidence": confidence,
            "iou_threshold": iou_threshold,
            "max_det": max_det,
        },
        "contract": {
            "dataset_audit_sha256": _sha256(grouped_audit),
            "grouped_manifest_sha256": _sha256(grouped_manifest),
            "d0_result_sha256": _sha256(d0_result),
            "d0_checkpoint_sha256": _sha256(d0_checkpoint),
            "reference_gates": reference["gates"],
            "grouped_audit_pass": audit.get("decision")
            == "PASS_GROUPED_DATASET_GATE",
            "test_images_accessed": False,
        },
        "manifest": manifest_stats,
        "rejected_groups": dict(rejected),
        "eligible_group_fraction": eligible_fraction,
        "reference_worst_class": worst_class,
        "reference_worst_class_complementary_groups": worst_complementarity,
        "summary": summary,
        "criteria": criteria,
        "decision": decision,
        "next": (
            "FREEZE_FRESH_DUAL_VIEW_MODEL_PROTOCOL"
            if scientific
            else "DO_NOT_TRAIN_DUAL_VIEW_MODEL"
        ),
        "claim_boundary": (
            "paired validation diagnostic; pair oracle uses ground truth only as an upper bound; "
            "no detector or fusion model was trained"
        ),
        "training_executed": False,
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if records_output is not None:
        record_path = Path(records_output).expanduser().resolve()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="DefectosCafeVerde dual-view audit")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-manifest", required=True)
    parser.add_argument("--grouped-audit", required=True)
    parser.add_argument("--d0-result", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--records-output")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_dual_view_audit(
        args.data_root,
        args.grouped_manifest,
        args.grouped_audit,
        args.d0_result,
        args.d0_checkpoint,
        args.output,
        records_output=args.records_output,
        device=args.device,
    )
    print("MANIFEST:", result["manifest"])
    print("REJECTED:", result["rejected_groups"])
    print("SUMMARY:", result["summary"])
    print("CRITERIA:", result["criteria"])
    print("DECISION:", result["decision"])
    print("NEXT:", result["next"])
    print("TRAINING:", result["training_executed"], "| TEST:", result["test_images_accessed"])


if __name__ == "__main__":
    main()
