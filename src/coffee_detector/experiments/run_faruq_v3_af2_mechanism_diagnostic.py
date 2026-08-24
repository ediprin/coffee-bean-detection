from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from coffee_detector.analysis.faruq_v3_diagnostics import run_faruq_v3_diagnostics
from coffee_detector.experiments.run_faruq_v3_baseline import (
    load_faruq_grouped_summary,
)


SEEDS = (42, 123, 2026)
MODELS = ("D0FT", "AF2")
MIN_MECHANISM_GAIN = 0.005
MIN_IMPROVED_SEEDS = 2


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_report(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("evaluation_split") != "val":
        raise RuntimeError(f"Diagnostic bukan validation: {path}")
    if payload.get("test_images_accessed") is not False:
        raise RuntimeError(f"Diagnostic tidak membuktikan test terkunci: {path}")
    if payload.get("training_executed") is not False:
        raise RuntimeError(f"Diagnostic tidak boleh menjalankan training: {path}")
    if "global" not in payload or "raw_candidate_sensitivity" not in payload:
        raise RuntimeError(f"Diagnostic tidak lengkap: {path}")
    return payload


def _metrics(report: dict) -> dict[str, float]:
    global_row = report["global"]
    raw = report["raw_candidate_sensitivity"]["500"]
    targets = max(int(global_row["targets"]), 1)
    matched = max(int(global_row["matched"]), 1)
    return {
        "raw_proposal_accessibility": float(raw["proposal_accessibility"]),
        "final_proposal_accessibility": float(
            global_row["proposal_accessibility"]
        ),
        "matched_recall": float(global_row["matched_recall"]),
        "conditional_top1_accuracy": float(
            global_row["localization_conditioned_class_accuracy"]
        ),
        "localized_wrong_class_rate": float(global_row["wrong_class"] / matched),
        "proposal_miss_rate": float(
            1.0 - global_row["proposal_accessibility"]
        ),
        "correct_decision_recall": float(global_row["correct_class"] / targets),
    }


def _aggregate(values: dict[int, dict[str, float]], metric: str) -> dict:
    controls = [values[seed]["D0FT"] for seed in SEEDS]
    candidates = [values[seed]["AF2"] for seed in SEEDS]
    deltas = {
        str(seed): float(values[seed]["AF2"] - values[seed]["D0FT"])
        for seed in SEEDS
    }
    delta_values = list(deltas.values())
    return {
        "d0ft_mean": float(statistics.mean(controls)),
        "af2_mean": float(statistics.mean(candidates)),
        "delta_mean": float(statistics.mean(delta_values)),
        "delta_std": float(statistics.pstdev(delta_values)),
        "delta_min": float(min(delta_values)),
        "improved_seeds": int(sum(value > 0.0 for value in delta_values)),
        "deltas": deltas,
    }


def _per_class_summary(reports: dict[int, dict[str, dict]]) -> list[dict]:
    names = {
        row["class_name"]
        for seed in SEEDS
        for model in MODELS
        for row in reports[seed][model]["per_class"]
    }
    output = []
    for name in sorted(names):
        paired = {}
        for seed in SEEDS:
            rows = {
                model: {
                    row["class_name"]: row
                    for row in reports[seed][model]["per_class"]
                }[name]
                for model in MODELS
            }
            paired[str(seed)] = {
                "proposal_accessibility_delta": float(
                    rows["AF2"]["proposal_accessibility"]
                    - rows["D0FT"]["proposal_accessibility"]
                ),
                "conditional_top1_accuracy_delta": float(
                    rows["AF2"]["localization_conditioned_class_accuracy"]
                    - rows["D0FT"]["localization_conditioned_class_accuracy"]
                ),
                "correct_decision_recall_delta": float(
                    rows["AF2"]["correct_class"]
                    / max(rows["AF2"]["targets"], 1)
                    - rows["D0FT"]["correct_class"]
                    / max(rows["D0FT"]["targets"], 1)
                ),
            }
        output.append(
            {
                "class_name": name,
                "mean_proposal_accessibility_delta": float(
                    statistics.mean(
                        row["proposal_accessibility_delta"]
                        for row in paired.values()
                    )
                ),
                "mean_conditional_top1_accuracy_delta": float(
                    statistics.mean(
                        row["conditional_top1_accuracy_delta"]
                        for row in paired.values()
                    )
                ),
                "mean_correct_decision_recall_delta": float(
                    statistics.mean(
                        row["correct_decision_recall_delta"]
                        for row in paired.values()
                    )
                ),
                "per_seed": paired,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["mean_conditional_top1_accuracy_delta"],
            row["mean_proposal_accessibility_delta"],
            row["class_name"],
        ),
    )


def build_af2_mechanism_summary(
    d0ft_reports: list[str | Path],
    af2_reports: list[str | Path],
    output: str | Path,
) -> dict:
    if len(d0ft_reports) != len(SEEDS) or len(af2_reports) != len(SEEDS):
        raise ValueError("D0FT dan AF2 masing-masing memerlukan tiga report")
    reports = {
        seed: {
            "D0FT": _load_report(d0ft_reports[index]),
            "AF2": _load_report(af2_reports[index]),
        }
        for index, seed in enumerate(SEEDS)
    }
    metric_rows = {
        seed: {model: _metrics(reports[seed][model]) for model in MODELS}
        for seed in SEEDS
    }
    per_seed = {
        str(seed): {
            model: metric_rows[seed][model] for model in MODELS
        }
        | {
            "deltas": {
                metric: float(
                    metric_rows[seed]["AF2"][metric]
                    - metric_rows[seed]["D0FT"][metric]
                )
                for metric in metric_rows[seed]["D0FT"]
            }
        }
        for seed in SEEDS
    }
    aggregate = {
        metric: _aggregate(
            {
                seed: {
                    model: metric_rows[seed][model][metric] for model in MODELS
                }
                for seed in SEEDS
            },
            metric,
        )
        for metric in metric_rows[SEEDS[0]]["D0FT"]
    }
    localization = aggregate["raw_proposal_accessibility"]
    classification = aggregate["conditional_top1_accuracy"]
    localization_supported = bool(
        localization["delta_mean"] >= MIN_MECHANISM_GAIN
        and localization["improved_seeds"] >= MIN_IMPROVED_SEEDS
    )
    classification_supported = bool(
        classification["delta_mean"] >= MIN_MECHANISM_GAIN
        and classification["improved_seeds"] >= MIN_IMPROVED_SEEDS
    )
    if localization_supported and classification_supported:
        attribution = "JOINT_LOCALIZATION_AND_CLASSIFICATION"
    elif classification_supported:
        attribution = "CLASSIFICATION_DOMINANT"
    elif localization_supported:
        attribution = "LOCALIZATION_DOMINANT"
    else:
        attribution = "MIXED_OR_UNRESOLVED"
    payload = {
        "format": "coffee_detector.af2.mechanism_diagnostic.v1",
        "protocol": "faruq-v3-af2-mechanism-diagnostic-v1",
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "training_executed": False,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "settings": {
            "image_size": 640,
            "matching_iou": 0.50,
            "confidence_threshold": 0.25,
            "raw_candidate_count": 500,
            "minimum_mechanism_gain": MIN_MECHANISM_GAIN,
            "minimum_improved_seeds": MIN_IMPROVED_SEEDS,
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "criteria": {
            "raw_localization_mean_gain_at_least_0_5_point": (
                localization["delta_mean"] >= MIN_MECHANISM_GAIN
            ),
            "raw_localization_improved_at_least_2_of_3": (
                localization["improved_seeds"] >= MIN_IMPROVED_SEEDS
            ),
            "conditional_classification_mean_gain_at_least_0_5_point": (
                classification["delta_mean"] >= MIN_MECHANISM_GAIN
            ),
            "conditional_classification_improved_at_least_2_of_3": (
                classification["improved_seeds"] >= MIN_IMPROVED_SEEDS
            ),
        },
        "localization_supported": localization_supported,
        "classification_supported": classification_supported,
        "attribution": attribution,
        "per_class": _per_class_summary(reports),
        "claim_limit": (
            "post-hoc validation association; not causal proof and not a model-selection gate"
        ),
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    payload["summary"] = str(destination)
    return payload


def run_af2_mechanism_diagnostic(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0ft_checkpoints: list[str | Path],
    af2_checkpoints: list[str | Path],
    output_root: str | Path,
    *,
    device: str = "cpu",
) -> dict:
    if len(d0ft_checkpoints) != len(SEEDS) or len(af2_checkpoints) != len(SEEDS):
        raise ValueError("D0FT dan AF2 masing-masing memerlukan tiga checkpoint")
    root = Path(data_root).expanduser().resolve()
    if (root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos split test")
    load_faruq_grouped_summary(grouped_summary, root)
    checkpoints = {
        seed: {
            "D0FT": Path(d0ft_checkpoints[index]).expanduser().resolve(),
            "AF2": Path(af2_checkpoints[index]).expanduser().resolve(),
        }
        for index, seed in enumerate(SEEDS)
    }
    for seed in SEEDS:
        for model in MODELS:
            if not checkpoints[seed][model].is_file():
                raise FileNotFoundError(checkpoints[seed][model])

    destination = Path(output_root).expanduser().resolve()
    reports_root = destination / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.af2.mechanism_contract.v1",
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "grouped_summary_sha256": _sha256(grouped_summary),
        "checkpoints": {
            str(seed): {
                model: {
                    "path": str(checkpoints[seed][model]),
                    "sha256": _sha256(checkpoints[seed][model]),
                }
                for model in MODELS
            }
            for seed in SEEDS
        },
        "training_executed": False,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    contract_path = destination / "input_contract.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError("Kontrak input diagnostic berubah; gunakan output baru")
    else:
        contract_path.write_text(
            json.dumps(contract, indent=2) + "\n", encoding="utf-8"
        )

    paths: dict[str, list[Path]] = {model: [] for model in MODELS}
    for index, seed in enumerate(SEEDS):
        for model in MODELS:
            report_path = reports_root / f"{model}_seed{seed}_diagnostic.json"
            if not report_path.is_file():
                print(f"DIAGNOSE {model} seed {seed}", flush=True)
                run_faruq_v3_diagnostics(
                    checkpoints[seed][model],
                    root,
                    report_path,
                    split="val",
                    device=device,
                )
            _load_report(report_path)
            paths[model].append(report_path)
            print(f"READY {model} seed {seed}: {report_path}", flush=True)

    return build_af2_mechanism_summary(
        paths["D0FT"],
        paths["AF2"],
        destination / "af2_mechanism_diagnostic.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only paired AF2 localization/classification diagnostic"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0ft-checkpoints", nargs=3, required=True)
    parser.add_argument("--af2-checkpoints", nargs=3, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_af2_mechanism_diagnostic(
        args.data_root,
        args.grouped_summary,
        args.d0ft_checkpoints,
        args.af2_checkpoints,
        args.output_root,
        device=args.device,
    )
    print(json.dumps({
        "attribution": result["attribution"],
        "localization_supported": result["localization_supported"],
        "classification_supported": result["classification_supported"],
        "criteria": result["criteria"],
    }, indent=2))
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
