"""Retracted audit for J25 identity-independent fixed-holdout support.

This audit is model-free.  Its primary unit is the recoverable source identity,
not the number of boxes.  A class represented by only one source identity in a
held-out split cannot estimate variation across independent captures, even when
that identity contains many annotated objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.data.prepare_coffee_standard_primary import J25_CLASSES
from coffee_detector.data.prepare_coffee_standard_primary_v2 import FORMAT


AUDIT_FORMAT = "coffee_detector.coffee_standard_j25_split_feasibility.v1"
RETRACTION = (
    "RETRACTED: stripped Roboflow filenames are not authoritative source identities in J25 v8. "
    "The thesis and export arithmetic establish 451 source images, while basename grouping produced 267 "
    "components by merging unrelated photographs. This audit must not be used."
)


def _load_json(path: str | Path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def audit_j25_split_feasibility(
    components_path: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
    output: str | Path,
    *,
    minimum_heldout_identities: int = 2,
    recommended_heldout_identities: int = 5,
    minimum_heldout_objects: int = 10,
) -> dict:
    """Prove whether a three-way identity-safe J25 holdout is supportable.

    The hard identity threshold defaults to two because this is the weakest
    non-degenerate held-out design: one identity alone cannot reveal
    between-capture variability.  Five identities is reported as a recommended
    planning target, not as a universal statistical law.
    """
    raise RuntimeError(RETRACTION)
    if minimum_heldout_identities < 2:  # pragma: no cover - historical code below is intentionally unreachable
        raise ValueError("minimum_heldout_identities harus minimal 2")
    if recommended_heldout_identities < minimum_heldout_identities:
        raise ValueError("recommended_heldout_identities tidak boleh lebih kecil dari hard gate")
    if minimum_heldout_objects < 1:
        raise ValueError("minimum_heldout_objects harus positif")

    components = _load_json(components_path)
    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path)
    if summary.get("format") != FORMAT:
        raise RuntimeError("Summary bukan Coffee Standard J25 v2")
    if summary.get("training_authorized") is not False:
        raise RuntimeError("Audit hanya menerima artifact yang training-nya masih terkunci")
    if len(components) != len(manifest):
        raise RuntimeError("Jumlah component dan manifest tidak sama")

    component_by_id = {row["component_id"]: row for row in components}
    if len(component_by_id) != len(components):
        raise RuntimeError("component_id tidak unik")
    split_by_id = {row["component_id"]: row["output_split"] for row in manifest}
    if set(split_by_id) != set(component_by_id):
        raise RuntimeError("Manifest dan component set tidak identik")
    if set(split_by_id.values()) != {"train", "val", "test"}:
        raise RuntimeError("Manifest harus memiliki train, val, dan test")

    rows = []
    for class_id, class_name in enumerate(J25_CLASSES):
        positive = [row for row in components if int(row["class_counts"][class_id]) > 0]
        total_identities = len(positive)
        total_objects = sum(int(row["class_counts"][class_id]) for row in positive)
        identities_by_split = {
            split: sum(split_by_id[row["component_id"]] == split for row in positive)
            for split in ("train", "val", "test")
        }
        objects_by_split = {
            split: sum(
                int(row["class_counts"][class_id])
                for row in positive
                if split_by_id[row["component_id"]] == split
            )
            for split in ("train", "val", "test")
        }
        # With at least one identity reserved for train, this is a mathematical
        # upper bound independent of any optimizer or random seed.
        common_heldout_identity_upper_bound = max(0, (total_identities - 1) // 2)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "total_objects": total_objects,
                "total_source_identities": total_identities,
                "current_identities_by_split": identities_by_split,
                "current_objects_by_split": objects_by_split,
                "current_minimum_heldout_identities": min(
                    identities_by_split["val"], identities_by_split["test"]
                ),
                "current_minimum_heldout_objects": min(
                    objects_by_split["val"], objects_by_split["test"]
                ),
                "theoretical_maximum_common_heldout_identities": common_heldout_identity_upper_bound,
                "hard_identity_gate_possible": common_heldout_identity_upper_bound
                >= minimum_heldout_identities,
                "recommended_identity_target_possible": common_heldout_identity_upper_bound
                >= recommended_heldout_identities,
                "current_object_gate_pass": min(
                    objects_by_split["val"], objects_by_split["test"]
                )
                >= minimum_heldout_objects,
            }
        )

    global_identity_ceiling = min(
        row["theoretical_maximum_common_heldout_identities"] for row in rows
    )
    limiting = [
        row["class_name"]
        for row in rows
        if row["theoretical_maximum_common_heldout_identities"] == global_identity_ceiling
    ]
    impossible = [row["class_name"] for row in rows if not row["hard_identity_gate_possible"]]
    current_identity_failures = [
        row["class_name"]
        for row in rows
        if row["current_minimum_heldout_identities"] < minimum_heldout_identities
    ]
    current_object_failures = [
        row["class_name"] for row in rows if not row["current_object_gate_pass"]
    ]
    gates = {
        "v2_artifact_contract_valid": True,
        "all_25_classes_have_train_val_test_objects": all(
            min(row["current_objects_by_split"].values()) > 0 for row in rows
        ),
        "current_split_meets_minimum_heldout_objects": not current_object_failures,
        "current_split_meets_minimum_heldout_identities": not current_identity_failures,
        "any_identity_safe_three_way_split_can_meet_hard_identity_gate": not impossible,
        "provenance_resolved": False,
        "training_not_executed": True,
        "model_test_evaluation_not_executed": True,
    }
    decision = (
        "PASS_FIXED_HOLDOUT_FEASIBILITY"
        if gates["any_identity_safe_three_way_split_can_meet_hard_identity_gate"]
        else "FAIL_FIXED_HOLDOUT_PRIMARY_BENCHMARK"
    )
    payload = {
        "format": AUDIT_FORMAT,
        "status": "complete",
        "dataset_format": summary["format"],
        "eligible_source_identities": len(components),
        "class_count": len(J25_CLASSES),
        "thresholds": {
            "minimum_heldout_objects_per_class": minimum_heldout_objects,
            "minimum_independent_identities_per_heldout_class": minimum_heldout_identities,
            "recommended_independent_identities_per_heldout_class": recommended_heldout_identities,
            "minimum_train_identities_per_class_for_upper_bound": 1,
        },
        "global_theoretical_maximum_common_heldout_identities": global_identity_ceiling,
        "limiting_classes": limiting,
        "classes_that_make_hard_identity_gate_impossible": impossible,
        "current_identity_gate_failures": current_identity_failures,
        "current_object_gate_failures": current_object_failures,
        "classes": rows,
        "gates": gates,
        "decision": decision,
        "next_action": (
            "USE_GROUPED_CROSS_VALIDATION_OR_ACQUIRE_MORE_INDEPENDENT_IDENTITIES"
            if decision.startswith("FAIL")
            else "FREEZE_V3_SPLIT_OPTIMIZATION_PROTOCOL"
        ),
        "interpretation": (
            "Box counts can be large while independent source support remains one. "
            "The identity ceiling is a mathematical bound, not an optimizer failure."
        ),
        "training_authorized": False,
        "test_access_authorized": False,
        "training_executed": False,
        "test_images_accessed_by_model": False,
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit J25 fixed-holdout feasibility by source identity.")
    parser.add_argument("--components", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-heldout-identities", type=int, default=2)
    parser.add_argument("--recommended-heldout-identities", type=int, default=5)
    parser.add_argument("--minimum-heldout-objects", type=int, default=10)
    args = parser.parse_args()
    result = audit_j25_split_feasibility(
        args.components,
        args.manifest,
        args.dataset_summary,
        args.output,
        minimum_heldout_identities=args.minimum_heldout_identities,
        recommended_heldout_identities=args.recommended_heldout_identities,
        minimum_heldout_objects=args.minimum_heldout_objects,
    )
    print("IDENTITY CEILING:", result["global_theoretical_maximum_common_heldout_identities"])
    print("LIMITING CLASSES:", result["limiting_classes"])
    print("CURRENT OBJECT FAILURES:", result["current_object_gate_failures"])
    print("CURRENT IDENTITY FAILURES:", result["current_identity_gate_failures"])
    print("GATES:", result["gates"])
    print("DECISION:", result["decision"])
    print("NEXT:", result["next_action"])
    print("TRAINING AUTHORIZED:", result["training_authorized"])
    print("SUMMARY:", result["summary"])


if __name__ == "__main__":
    main()
