from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from coffee_detector.analysis.faruq_v3_diagnostics import run_faruq_v3_diagnostics
from coffee_detector.dataset import write_json
from coffee_detector.sni21_ontology import load_sni21_ontology


TASKS = (
    "entity_family",
    "primary_condition",
    "hole_count",
    "integrity_fraction",
    "surface_extent",
)


def _pair_category(expected: dict, predicted: dict) -> str:
    if expected["primary_condition"] == predicted["primary_condition"]:
        return "same_primary_condition"
    if expected["entity_family"] == predicted["entity_family"]:
        return "same_entity_family_only"
    return "cross_entity_family"


def attribute_directional_confusions(confusion: dict, ontology: dict) -> dict:
    categories = Counter()
    task_agreement = Counter()
    pair_rows = []
    total_wrong = 0
    classes = ontology["classes"]
    for expected_name, predictions in confusion.items():
        for predicted_name, raw_count in predictions.items():
            count = int(raw_count)
            if expected_name == predicted_name or count <= 0:
                continue
            expected = classes[expected_name]
            predicted = classes[predicted_name]
            category = _pair_category(expected, predicted)
            same_tasks = [
                task
                for task in TASKS
                if expected.get(task) is not None
                and expected.get(task) == predicted.get(task)
            ]
            categories[category] += count
            total_wrong += count
            for task in same_tasks:
                task_agreement[task] += count
            pair_rows.append(
                {
                    "expected": expected_name,
                    "predicted": predicted_name,
                    "count": count,
                    "category": category,
                    "same_tasks": same_tasks,
                }
            )
    return {
        "wrong_class": total_wrong,
        "category_counts": dict(categories),
        "category_fractions": {
            category: count / max(total_wrong, 1)
            for category, count in categories.items()
        },
        "same_task_counts": dict(task_agreement),
        "same_task_fractions": {
            task: count / max(total_wrong, 1)
            for task, count in task_agreement.items()
        },
        "pairs": sorted(
            pair_rows,
            key=lambda row: (-row["count"], row["expected"], row["predicted"]),
        ),
    }


def _pair_deltas(candidate: list[dict], baseline: list[dict]) -> list[dict]:
    candidate_counts = {
        (row["expected"], row["predicted"]): int(row["count"]) for row in candidate
    }
    baseline_counts = {
        (row["expected"], row["predicted"]): int(row["count"]) for row in baseline
    }
    rows = []
    for key in sorted(set(candidate_counts) | set(baseline_counts)):
        delta = candidate_counts.get(key, 0) - baseline_counts.get(key, 0)
        if delta:
            rows.append(
                {
                    "expected": key[0],
                    "predicted": key[1],
                    "baseline_count": baseline_counts.get(key, 0),
                    "candidate_count": candidate_counts.get(key, 0),
                    "delta": delta,
                }
            )
    return sorted(rows, key=lambda row: (-row["delta"], row["expected"], row["predicted"]))


def _diagnostic(
    code: str,
    checkpoint: Path,
    data_root: Path,
    output_root: Path,
    device: str,
) -> dict:
    destination = output_root / f"{code}_seed42_full_diagnostic.json"
    if destination.is_file():
        cached = json.loads(destination.read_text(encoding="utf-8"))
        if (
            cached.get("evaluation_split") == "val"
            and cached.get("test_images_accessed") is False
            and "directional_confusions" in cached
        ):
            print(f"REUSE FULL CONFUSION: {code}", flush=True)
            return cached
    return run_faruq_v3_diagnostics(
        checkpoint, data_root, destination, split="val", device=device
    )


def run_ontology_error_attribution(
    data_root: str | Path,
    d0_checkpoint: str | Path,
    c0_checkpoint: str | Path,
    s0_checkpoint: str | Path,
    output_root: str | Path,
    *,
    device: str = "cpu",
) -> dict:
    data_root = Path(data_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Ontology error attribution tidak boleh menyediakan test")
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "D0": Path(d0_checkpoint).expanduser().resolve(),
        "C0": Path(c0_checkpoint).expanduser().resolve(),
        "S0": Path(s0_checkpoint).expanduser().resolve(),
    }
    for code, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint {code} tidak ditemukan: {checkpoint}")

    ontology = load_sni21_ontology()
    diagnostics = {
        code: _diagnostic(code, checkpoint, data_root, output_root, device)
        for code, checkpoint in checkpoints.items()
    }
    models = {}
    for code, diagnostic in diagnostics.items():
        attribution = attribute_directional_confusions(
            diagnostic["directional_confusions"], ontology
        )
        recorded_wrong = int(diagnostic["global"]["wrong_class"])
        if attribution["wrong_class"] != recorded_wrong:
            raise RuntimeError(
                f"Confusion {code} tidak lengkap: {attribution['wrong_class']} != {recorded_wrong}"
            )
        models[code] = {
            "global": diagnostic["global"],
            **attribution,
        }

    category_delta = {
        category: models["S0"]["category_counts"].get(category, 0)
        - models["D0"]["category_counts"].get(category, 0)
        for category in (
            "same_primary_condition",
            "same_entity_family_only",
            "cross_entity_family",
        )
    }
    within_delta = (
        category_delta["same_primary_condition"]
        + category_delta["same_entity_family_only"]
    )
    outside_delta = category_delta["cross_entity_family"]
    interpretation = (
        "ONTOLOGY_COARSENING_DOMINATES"
        if within_delta > outside_delta
        else "OUTSIDE_GROUP_ERRORS_DOMINATE"
    )
    payload = {
        "protocol": "faruq-v3-ontology-error-attribution-v1",
        "evaluation_split": "val",
        "training_executed": False,
        "test_images_accessed": False,
        "models": models,
        "S0_vs_D0": {
            "category_count_deltas": category_delta,
            "within_ontology_delta": within_delta,
            "outside_ontology_delta": outside_delta,
            "largest_pair_increases": _pair_deltas(
                models["S0"]["pairs"], models["D0"]["pairs"]
            )[:15],
        },
        "interpretation": interpretation,
        "next_action": (
            "design_leaf_preserving_objective_before_any_training"
            if interpretation == "ONTOLOGY_COARSENING_DOMINATES"
            else "stop_structured_ontology_direction"
        ),
    }
    destination = output_root / "ontology_error_attribution.json"
    write_json(payload, destination)
    payload["summary"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute D0/C0/S0 errors to SNI ontology")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--c0-checkpoint", required=True)
    parser.add_argument("--s0-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_ontology_error_attribution(
        args.data_root,
        args.d0_checkpoint,
        args.c0_checkpoint,
        args.s0_checkpoint,
        args.output_root,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
