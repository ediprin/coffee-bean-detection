from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.analysis.coffee_fg_diagnostics import diagnose_checkpoint


def _top_confusions(confusion: dict[str, dict[str, int]], limit: int = 15) -> list[dict]:
    rows = []
    for expected, predictions in confusion.items():
        for predicted, count in predictions.items():
            if expected != predicted:
                rows.append(
                    {"expected": expected, "predicted": predicted, "count": int(count)}
                )
    return sorted(rows, key=lambda row: (-row["count"], row["expected"], row["predicted"]))[:limit]


def run_faruq_v3_diagnostics(
    checkpoint: str | Path,
    data_root: str | Path,
    output: str | Path,
    *,
    split: str = "val",
    device: str = "cpu",
) -> dict:
    if split != "val":
        raise RuntimeError("Faruq-v3 diagnostic dikunci pada validation")
    diagnostic = diagnose_checkpoint(
        checkpoint,
        data_root,
        split="val",
        image_size=640,
        candidate_counts=(50, 100, 300, 500),
        iou_threshold=0.5,
        confidence_threshold=0.25,
        nms_iou=0.7,
        max_det=500,
        device=device,
    )
    selected = diagnostic["final_detections"]
    class_rows = [
        {"class_name": name, **metrics}
        for name, metrics in selected["per_class"].items()
    ]
    payload = {
        "protocol": "faruq-v3-yolo26n-diagnostic-v1",
        "training_executed": False,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "global": {
            key: selected[key]
            for key in (
                "targets",
                "accessible",
                "matched",
                "correct_class",
                "wrong_class",
                "missed",
                "proposal_accessibility",
                "matched_recall",
                "localization_conditioned_class_accuracy",
                "oracle_class_accuracy_headroom",
            )
        },
        "per_class": sorted(
            class_rows,
            key=lambda row: (
                row["localization_conditioned_class_accuracy"],
                row["proposal_accessibility"],
                row["class_name"],
            ),
        ),
        "top_directional_confusions": _top_confusions(selected["confusion"]),
        "raw_candidate_sensitivity": {
            count: {
                key: diagnostic["branches"]["one2one"][count][key]
                for key in (
                    "proposal_accessibility",
                    "matched_recall",
                    "localization_conditioned_class_accuracy",
                )
            }
            for count in ("50", "100", "300", "500")
        },
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
        description="Validation-only proposal/localization/classification audit for Faruq-v3."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_faruq_v3_diagnostics(
        args.checkpoint,
        args.data_root,
        args.output,
        split=args.split,
        device=args.device,
    )
    print(json.dumps(result["global"], indent=2, ensure_ascii=False))
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
