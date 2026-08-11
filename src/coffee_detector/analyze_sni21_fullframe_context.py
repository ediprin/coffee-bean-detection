from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .prepare_sni_fullscene import SNI21_CLASSES
from .run_sni21_density_evaluation import _pairwise_iou


ARMS = (
    "FC1_repaste_real_fullframe",
    "FC2_repaste_procedural_fullframe",
)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Bukan JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        image = str(row["image"])
        if image in rows:
            raise RuntimeError(f"Duplikat image pada {path}:{line_number}: {image}")
        rows[image] = row
    return rows


def corrected_record_diagnosis(record: dict, iou_threshold: float) -> dict:
    ground_truth = record.get("ground_truth", [])
    if len(ground_truth) != 1:
        raise RuntimeError(
            f"Kontrol full-frame harus memiliki satu GT: {record.get('image')}"
        )
    gt = ground_truth[0]
    predictions = record.get("predictions", [])
    if predictions:
        pred_boxes = np.asarray(
            [row["xyxy"] for row in predictions], dtype=np.float64
        ).reshape(-1, 4)
        gt_box = np.asarray([gt["xyxy"]], dtype=np.float64)
        ious = _pairwise_iou(pred_boxes, gt_box)[:, 0]
        localized = [
            index for index, value in enumerate(ious) if value >= iou_threshold
        ]
    else:
        ious = np.empty(0, dtype=np.float64)
        localized = []
    if not localized:
        return {
            "class_id": int(gt["class_id"]),
            "state": "miss",
            "proposal": 0,
            "correct": 0,
            "predicted_class": None,
            "confidence": None,
            "iou": float(ious.max()) if len(ious) else 0.0,
        }
    selected = max(
        localized,
        key=lambda index: (
            float(predictions[index]["confidence"]),
            float(ious[index]),
        ),
    )
    predicted_class = int(predictions[selected]["class_id"])
    correct = int(predicted_class == int(gt["class_id"]))
    return {
        "class_id": int(gt["class_id"]),
        "state": "correct" if correct else "wrong",
        "proposal": 1,
        "correct": correct,
        "predicted_class": predicted_class,
        "confidence": float(predictions[selected]["confidence"]),
        "iou": float(ious[selected]),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _macro(rows: list[dict], field: str) -> float:
    by_class: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_class[int(row["class_id"])].append(float(row[field]))
    return _mean([_mean(values) for values in by_class.values()])


def _bootstrap(
    paired: list[dict], *, iterations: int, seed: int
) -> dict:
    if iterations <= 0:
        raise ValueError("iterations harus positif")
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in paired:
        grouped[int(row["class_id"])].append(row)
    rng = np.random.default_rng(seed)
    accuracy_deltas = np.empty(iterations, dtype=np.float64)
    proposal_deltas = np.empty(iterations, dtype=np.float64)
    groups = [grouped[class_id] for class_id in sorted(grouped)]
    for iteration in range(iterations):
        class_accuracy = []
        class_proposal = []
        for rows in groups:
            indices = rng.integers(0, len(rows), size=len(rows))
            sampled = [rows[int(index)] for index in indices]
            class_accuracy.append(
                _mean([row["fc2_correct"] - row["fc1_correct"] for row in sampled])
            )
            class_proposal.append(
                _mean([row["fc2_proposal"] - row["fc1_proposal"] for row in sampled])
            )
        accuracy_deltas[iteration] = _mean(class_accuracy)
        proposal_deltas[iteration] = _mean(class_proposal)

    def summarize(values: np.ndarray, point: float) -> dict:
        return {
            "point": point,
            "ci95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "probability_below_zero": float(np.mean(values < 0)),
            "iterations": iterations,
            "seed": seed,
        }

    accuracy_point = _macro(paired, "fc2_correct") - _macro(
        paired, "fc1_correct"
    )
    proposal_point = _macro(paired, "fc2_proposal") - _macro(
        paired, "fc1_proposal"
    )
    return {
        "macro_top1_accuracy_delta": summarize(
            accuracy_deltas, accuracy_point
        ),
        "macro_proposal_recall_delta": summarize(
            proposal_deltas, proposal_point
        ),
    }


def _exact_mcnemar(harmed: int, rescued: int) -> dict:
    discordant = harmed + rescued
    if discordant == 0:
        return {"harmed": harmed, "rescued": rescued, "discordant": 0, "p_two_sided": 1.0}
    lower = min(harmed, rescued)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2**discordant)
    return {
        "harmed": harmed,
        "rescued": rescued,
        "discordant": discordant,
        "p_two_sided": min(1.0, 2.0 * tail),
    }


def analyze_fullframe_context(
    evaluation_root: str | Path,
    *,
    iou_threshold: float = 0.5,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict:
    root = Path(evaluation_root).expanduser().resolve()
    summary = _read_json(root / "local_context_summary.json")
    if summary.get("training_executed") is not False:
        raise RuntimeError("Summary tidak menjamin training=False")
    if summary.get("test_images_accessed") is not False:
        raise RuntimeError("Summary telah mengakses test")

    records = {
        arm: _read_jsonl(root / arm / "prediction_records.jsonl") for arm in ARMS
    }
    if set(records[ARMS[0]]) != set(records[ARMS[1]]):
        raise RuntimeError("FC1 dan FC2 tidak memiliki image identity yang sama")
    official = {
        arm: _read_json(root / arm / "evaluation.json")["official_metrics"]
        for arm in ARMS
    }

    paired = []
    transitions = Counter()
    per_class_rows: dict[int, list[dict]] = defaultdict(list)
    for image in sorted(records[ARMS[0]]):
        fc1 = corrected_record_diagnosis(records[ARMS[0]][image], iou_threshold)
        fc2 = corrected_record_diagnosis(records[ARMS[1]][image], iou_threshold)
        if fc1["class_id"] != fc2["class_id"]:
            raise RuntimeError(f"Ground truth berbeda antar-arm: {image}")
        row = {
            "image": image,
            "class_id": fc1["class_id"],
            "class_name": SNI21_CLASSES[fc1["class_id"]],
            "fc1_state": fc1["state"],
            "fc2_state": fc2["state"],
            "fc1_proposal": fc1["proposal"],
            "fc2_proposal": fc2["proposal"],
            "fc1_correct": fc1["correct"],
            "fc2_correct": fc2["correct"],
            "fc1_predicted_class": fc1["predicted_class"],
            "fc2_predicted_class": fc2["predicted_class"],
            "fc1_confidence": fc1["confidence"],
            "fc2_confidence": fc2["confidence"],
        }
        paired.append(row)
        per_class_rows[fc1["class_id"]].append(row)
        transitions[f"{fc1['state']}->{fc2['state']}"] += 1

    class_table = []
    for class_id, rows in sorted(per_class_rows.items()):
        name = SNI21_CLASSES[class_id]
        fc1_proposal = _mean([row["fc1_proposal"] for row in rows])
        fc2_proposal = _mean([row["fc2_proposal"] for row in rows])
        fc1_accuracy = _mean([row["fc1_correct"] for row in rows])
        fc2_accuracy = _mean([row["fc2_correct"] for row in rows])
        fc1_ap = official[ARMS[0]]["map50_95_by_class"].get(name)
        fc2_ap = official[ARMS[1]]["map50_95_by_class"].get(name)
        class_table.append(
            {
                "class_id": class_id,
                "class_name": name,
                "n": len(rows),
                "fc1_proposal_recall": fc1_proposal,
                "fc2_proposal_recall": fc2_proposal,
                "delta_proposal_recall": fc2_proposal - fc1_proposal,
                "fc1_top1_accuracy": fc1_accuracy,
                "fc2_top1_accuracy": fc2_accuracy,
                "delta_top1_accuracy": fc2_accuracy - fc1_accuracy,
                "fc1_map50_95": fc1_ap,
                "fc2_map50_95": fc2_ap,
                "delta_map50_95": (
                    fc2_ap - fc1_ap
                    if fc1_ap is not None and fc2_ap is not None
                    else None
                ),
            }
        )

    bootstrap = _bootstrap(paired, iterations=iterations, seed=seed)
    harmed = sum(
        row["fc1_correct"] == 1 and row["fc2_correct"] == 0 for row in paired
    )
    rescued = sum(
        row["fc1_correct"] == 0 and row["fc2_correct"] == 1 for row in paired
    )
    mcnemar = _exact_mcnemar(harmed, rescued)
    official_delta = (
        official[ARMS[1]]["metrics/mAP50-95(B)"]
        - official[ARMS[0]]["metrics/mAP50-95(B)"]
    )
    accuracy_bootstrap = bootstrap["macro_top1_accuracy_delta"]
    supported = (
        official_delta < 0
        and accuracy_bootstrap["ci95"][1] < 0
        and mcnemar["p_two_sided"] < 0.05
    )
    output = {
        "format": "coffee_detector.sni21_fullframe_context_paired_analysis.v1",
        "analysis_status": "posthoc_exploratory",
        "samples": len(paired),
        "classes": len(class_table),
        "iou_threshold": iou_threshold,
        "class_decision": "highest-confidence prediction among candidates with IoU >= threshold",
        "official_map50_95_delta_fc2_minus_fc1": official_delta,
        "micro_top1_accuracy": {
            "fc1": _mean([row["fc1_correct"] for row in paired]),
            "fc2": _mean([row["fc2_correct"] for row in paired]),
        },
        "macro_top1_accuracy": {
            "fc1": _macro(paired, "fc1_correct"),
            "fc2": _macro(paired, "fc2_correct"),
        },
        "macro_proposal_recall": {
            "fc1": _macro(paired, "fc1_proposal"),
            "fc2": _macro(paired, "fc2_proposal"),
        },
        "transitions": dict(sorted(transitions.items())),
        "paired_mcnemar_top1": mcnemar,
        "stratified_bootstrap": bootstrap,
        "paired_background_harm_supported": supported,
        "interpretation": (
            "procedural_background_harms_classification_on_this_development_subset"
            if supported
            else "paired_evidence_not_conclusive"
        ),
        "training_executed": False,
        "inference_executed": False,
        "test_images_accessed": False,
        "class_table": class_table,
    }
    csv_path = root / "paired_context_class_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_table[0]))
        writer.writeheader()
        writer.writerows(class_table)
    json_path = root / "paired_context_analysis.json"
    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== PAIRED FULL-FRAME CONTEXT ANALYSIS ===")
    print(f"Samples/classes : {len(paired)}/{len(class_table)}")
    print(f"Official Delta mAP: {official_delta:+.2%}")
    print("Macro top-1     :", output["macro_top1_accuracy"])
    print("Macro proposal  :", output["macro_proposal_recall"])
    print("Transitions     :", output["transitions"])
    print("McNemar         :", mcnemar)
    print("Bootstrap       :", bootstrap)
    print("SUPPORTED       :", supported)
    print("SAVED           :", json_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc paired analysis of SNI-21 full-frame context control"
    )
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze_fullframe_context(
        args.evaluation_root,
        iou_threshold=args.iou_threshold,
        iterations=args.iterations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
