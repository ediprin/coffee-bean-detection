from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from coffee_detector.dataset import discover_layout, parse_label, write_json
from coffee_detector.sni21_ontology import SIGNATURE_FIELDS, load_sni21_ontology


SCALAR_TASKS = ("entity_family", "primary_condition", *SIGNATURE_FIELDS[2:])
MIN_SUPPORT = {
    "train_instances": 50,
    "train_groups": 25,
    "val_instances": 10,
    "val_groups": 10,
}


def _load_group_manifest(data_root: Path) -> list[dict]:
    path = data_root / "faruq_grouped_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Manifest grouping tidak ditemukan: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        split = str(row["output_split"])
        image_name = Path(row["output_image"]).name.lower()
        key = (split, image_name)
        if key in seen and seen[key] != row["group_id"]:
            raise RuntimeError(f"Nama gambar ambigu pada manifest: {key}")
        seen[key] = str(row["group_id"])
    return rows


def _support_status(row: dict) -> tuple[bool, list[str]]:
    failures = [
        field
        for field, threshold in MIN_SUPPORT.items()
        if int(row[field]) < threshold
    ]
    return not failures, failures


def audit_sni21_structured_targets(
    data_root: str | Path,
    output: str | Path,
    *,
    ontology_path: str | Path | None = None,
) -> dict:
    """Audit statistical support and observability without training a model."""

    layout = discover_layout(data_root)
    if "val" not in layout.splits:
        raise FileNotFoundError("Validation split tidak ditemukan")
    if "test" in layout.splits or (layout.root / "test").exists():
        raise RuntimeError("Audit development tidak boleh menyediakan test")

    ontology = load_sni21_ontology(ontology_path)
    if tuple(layout.names[index] for index in sorted(layout.names)) != tuple(
        ontology["classes"]
    ):
        raise ValueError("Urutan kelas data.yaml tidak identik dengan ontologi SNI-21")
    manifest = _load_group_manifest(layout.root)

    counts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    images: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    totals = Counter()
    observed_totals = Counter()

    for manifest_row in manifest:
        split = str(manifest_row["output_split"])
        if split not in {"train", "val"}:
            raise RuntimeError(f"Manifest berisi split terlarang: {split}")
        _, label_root = layout.splits[split]
        image_name = Path(manifest_row["output_image"]).name
        label_path = label_root / Path(manifest_row["output_label"]).name
        group_id = str(manifest_row["group_id"])
        image_id = image_name
        manifest_counts = list(manifest_row.get("class_counts", []))
        parsed_boxes = parse_label(label_path, set(layout.names))
        if manifest_counts:
            parsed_counts = Counter(box.class_id for box in parsed_boxes)
            expected_counts = Counter(
                {index: int(count) for index, count in enumerate(manifest_counts) if count}
            )
            if parsed_counts != expected_counts:
                raise RuntimeError(f"Hitungan label tidak cocok dengan manifest: {label_path}")
        for box in parsed_boxes:
            class_name = layout.names[box.class_id]
            class_row = ontology["classes"][class_name]
            totals[split] += 1
            values: dict[str, list[str]] = {
                "original_class": [class_name],
                "positive_flag": [str(value) for value in class_row["positive_flags"]],
            }
            for task in SCALAR_TASKS:
                value = class_row.get(task)
                if value is not None:
                    values[task] = [str(value)]
                    observed_totals[(split, task)] += 1
            for task, task_values in values.items():
                for value in task_values:
                    item = (split, task, value)
                    counts[item]["instances"] += 1
                    images[item].add(image_id)
                    groups[item].add(group_id)

    value_rows = []
    task_values: dict[str, set[str]] = defaultdict(set)
    for _, task, value in counts:
        task_values[task].add(value)
    for task in sorted(task_values):
        for value in sorted(task_values[task]):
            row = {"task": task, "value": value}
            for split in ("train", "val"):
                item = (split, task, value)
                row[f"{split}_instances"] = counts[item]["instances"]
                row[f"{split}_images"] = len(images[item])
                row[f"{split}_groups"] = len(groups[item])
            supported, failures = _support_status(row)
            row["statistically_supported"] = supported
            row["support_failures"] = failures
            value_rows.append(row)

    observability = dict(ontology.get("observability", {}))
    observability.update(
        {
            "original_class": "rgb_observable_but_fine_grained",
            "positive_flag": "positive_only_partial_supervision",
        }
    )
    task_rows = []
    for task in sorted(task_values):
        rows = [row for row in value_rows if row["task"] == task]
        state = observability.get(task, "not_documented")
        if state == "calibrated_scale_required":
            semantic_gate = "BLOCKED_REQUIRES_CALIBRATED_SCALE"
        elif "reference_sensitive" in state:
            semantic_gate = "REQUIRES_DOMAIN_EXPERT_REVIEW"
        elif state == "not_documented":
            semantic_gate = "BLOCKED_OBSERVABILITY_UNDOCUMENTED"
        else:
            semantic_gate = "ELIGIBLE_FOR_PROTOCOL_DESIGN"
        observed_train = (
            totals["train"]
            if task in {"original_class", "positive_flag", "entity_family", "primary_condition"}
            else observed_totals[("train", task)]
        )
        observed_val = (
            totals["val"]
            if task in {"original_class", "positive_flag", "entity_family", "primary_condition"}
            else observed_totals[("val", task)]
        )
        task_rows.append(
            {
                "task": task,
                "values": len(rows),
                "observability": state,
                "train_observed_fraction": observed_train / max(1, totals["train"]),
                "val_observed_fraction": observed_val / max(1, totals["val"]),
                "minimum_train_instances": min(row["train_instances"] for row in rows),
                "minimum_train_groups": min(row["train_groups"] for row in rows),
                "minimum_val_instances": min(row["val_instances"] for row in rows),
                "minimum_val_groups": min(row["val_groups"] for row in rows),
                "all_values_statistically_supported": all(
                    row["statistically_supported"] for row in rows
                ),
                "semantic_gate": semantic_gate,
            }
        )

    statistically_ready = all(
        row["all_values_statistically_supported"]
        for row in task_rows
        if row["task"] not in {"positive_flag"}
    )
    blocked_tasks = [
        row["task"]
        for row in task_rows
        if row["semantic_gate"].startswith("BLOCKED")
    ]
    review_tasks = [
        row["task"]
        for row in task_rows
        if row["semantic_gate"] == "REQUIRES_DOMAIN_EXPERT_REVIEW"
    ]
    payload = {
        "protocol": "sni21-structured-target-support-audit-v1",
        "dataset_root": str(layout.root),
        "ontology_source": ontology["source"],
        "evaluation_splits": ["train", "val"],
        "training_executed": False,
        "inference_executed": False,
        "test_images_accessed": False,
        "test_locked": True,
        "thresholds": MIN_SUPPORT,
        "instances_by_split": dict(totals),
        "task_rows": task_rows,
        "value_rows": value_rows,
        "statistically_ready": statistically_ready,
        "blocked_tasks": blocked_tasks,
        "domain_expert_review_tasks": review_tasks,
        "decision": "AUDIT_COMPLETE_NO_TRAINING_AUTHORIZATION",
        "next_action": "review_unsupported_values_and_semantic_gates_before_freezing_model_protocol",
        "limitations": [
            "Counts inherited from leaf labels do not prove that an attribute is visually observable.",
            "Positive flags are partial supervision; unspecified flags are unknown rather than negative.",
            "Physical-size targets cannot be learned from pixels without a calibrated scale reference.",
            "Train/validation support is development evidence and does not authorize test access.",
        ],
    }
    write_json(payload, Path(output).expanduser().resolve())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SNI-21 structured-target support")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ontology")
    args = parser.parse_args()
    payload = audit_sni21_structured_targets(
        args.data_root, args.output, ontology_path=args.ontology
    )
    print(json.dumps({
        "decision": payload["decision"],
        "statistically_ready": payload["statistically_ready"],
        "blocked_tasks": payload["blocked_tasks"],
        "domain_expert_review_tasks": payload["domain_expert_review_tasks"],
        "output": str(Path(args.output).expanduser().resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
