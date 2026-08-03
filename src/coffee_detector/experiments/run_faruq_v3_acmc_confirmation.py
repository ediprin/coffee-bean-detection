"""Paired three-seed confirmation of the ACMC1 one-stage head on Faruq-v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.ambiguity_multilevel.audit import static_ambiguity_multilevel_audit
from coffee_detector.experiments.run_faruq_v3_acmc import run_faruq_v3_acmc
from coffee_detector.experiments.run_faruq_v3_baseline import run_faruq_v3_baseline


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
DEFAULT_SEEDS = (42, 123, 2026)


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_faruq_v3_acmc_confirmation(
    data_root: str | Path,
    grouped_summary: str | Path,
    baseline_root: str | Path,
    output_root: str | Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    """Train paired D0/ACMC runs only where a compatible checkpoint is missing."""
    frozen_seeds = tuple(int(seed) for seed in seeds)
    if frozen_seeds != DEFAULT_SEEDS:
        raise ValueError(f"Konfirmasi ACMC1 dikunci pada seed {DEFAULT_SEEDS}")
    if not authorize_training:
        raise RuntimeError("Konfirmasi ACMC1 belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    baseline_root = Path(baseline_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    reports_root = output_root / "val_reports"
    static_root = output_root / "static_audits"
    reports_root.mkdir(parents=True, exist_ok=True)
    static_root.mkdir(parents=True, exist_ok=True)

    per_seed: dict[str, dict] = {}
    for seed in frozen_seeds:
        print(f"\n=== ACMC1 CONFIRMATION | SEED {seed} ===", flush=True)
        baseline = run_faruq_v3_baseline(
            data_root, grouped_summary, baseline_root, seed=seed, device=device
        )
        d0_checkpoint = baseline_root / f"D0_seed{seed}" / "weights" / "best.pt"
        static_path = static_root / f"D0_seed{seed}_static.json"
        static = static_ambiguity_multilevel_audit(
            MODEL_YAML, d0_checkpoint, static_path, nc=21, image_size=128
        )
        if static["decision"] != "PASS":
            raise RuntimeError(f"Static ACMC D0 seed {seed} gagal: {static_path}")
        candidate = run_faruq_v3_acmc(
            data_root,
            grouped_summary,
            baseline["summary"],
            d0_checkpoint,
            static_path,
            output_root,
            seed=seed,
            device=device,
            authorize_training=True,
            confirmation_seed=(seed != 42),
        )
        per_seed[str(seed)] = {
            "d0_summary": baseline["summary"],
            "acmc_summary": candidate["summary"],
            "static_audit": str(static_path),
            "results": candidate["results"],
            "deltas": candidate["deltas"],
        }

    aggregate: dict[str, dict[str, float | int]] = {}
    for metric in METRICS:
        deltas = [float(per_seed[str(seed)]["deltas"][metric]) for seed in frozen_seeds]
        aggregate[metric] = {
            "d0_mean": sum(float(per_seed[str(seed)]["results"]["D0"][metric]) for seed in frozen_seeds)
            / len(frozen_seeds),
            "acmc1_mean": sum(float(per_seed[str(seed)]["results"]["ACMC1"][metric]) for seed in frozen_seeds)
            / len(frozen_seeds),
            "delta_mean": sum(deltas) / len(deltas),
            "delta_min": min(deltas),
            "improved_seeds": sum(delta > 0.0 for delta in deltas),
        }
    criteria = {
        "macro_mean_gain_at_least_0_5_point": aggregate["macro_map50_95"]["delta_mean"] >= 0.005,
        "macro_improved_at_least_2_of_3": aggregate["macro_map50_95"]["improved_seeds"] >= 2,
        "bottom3_mean_not_lower": aggregate["bottom3_class_map50_95"]["delta_mean"] >= 0.0,
        "bottom3_improved_at_least_2_of_3": aggregate["bottom3_class_map50_95"]["improved_seeds"] >= 2,
        "worst_mean_drop_no_more_than_1_point": aggregate["worst_class_map50_95"]["delta_mean"] >= -0.01,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    payload = {
        "protocol": "faruq-v3-acmc-one-stage-v1",
        "stage": "three_seed_confirmation",
        "seeds": list(frozen_seeds),
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": decision,
        "next_action": (
            "AUTHORIZE_SINGLE_LOCKED_TEST_EVALUATION" if decision == "PASS"
            else "STOP_ACMC1_WITHOUT_TEST"
        ),
    }
    summary = reports_root / "acmc1_three_seed_confirmation.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired Faruq-v3 ACMC1 confirmation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_acmc_confirmation(
        args.data_root,
        args.grouped_summary,
        args.baseline_root,
        args.output_root,
        seeds=tuple(args.seeds),
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
