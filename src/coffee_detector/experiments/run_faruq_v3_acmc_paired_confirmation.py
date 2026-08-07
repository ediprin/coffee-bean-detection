"""Optimization-matched three-seed confirmation for ACMC on Faruq-v3.

The original ACMC three-seed run compared each ACMC continuation against D0
*before* its extra continuation epochs.  This runner deliberately does not
reuse that result.  It uses the valid seed-42 control as one locked pair, then
creates two new, independent D0 -> {D0FT, ACMC1} pairs for seeds 123 and 2026.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.ambiguity_multilevel.audit import static_ambiguity_multilevel_audit
from coffee_detector.experiments.run_faruq_v3_acmc import METRICS, run_faruq_v3_acmc
from coffee_detector.experiments.run_faruq_v3_acmc_finetune_control import (
    _metrics,
    run_d0ft_continuation,
)
from coffee_detector.experiments.run_faruq_v3_baseline import run_faruq_v3_baseline


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
SEED42 = 42
CONFIRMATION_SEEDS = (123, 2026)


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _validate_seed42_control(path: str | Path) -> dict:
    payload = _load_json(path, "Seed-42 optimization control")
    if (
        payload.get("protocol") != "faruq-v3-acmc-optimization-control-v1"
        or int(payload.get("seed", -1)) != SEED42
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
        or payload.get("decision") != "PASS"
    ):
        raise RuntimeError("Seed-42 control bukan PASS validation-only yang kompatibel")
    for arm in ("D0", "D0FT", "ACMC1"):
        _metrics(payload, arm)
    return payload


def _seed_record(d0: dict, d0ft: dict, acmc: dict, static_audit: Path) -> dict:
    d0_metrics = _metrics(d0, "D0")
    d0ft_metrics = _metrics(d0ft)
    acmc_metrics = _metrics(acmc, "ACMC1")
    return {
        "results": {"D0": d0_metrics, "D0FT": d0ft_metrics, "ACMC1": acmc_metrics},
        "control_deltas_d0ft_vs_d0": {
            metric: d0ft_metrics[metric] - d0_metrics[metric] for metric in METRICS
        },
        "head_deltas_acmc1_vs_d0ft": {
            metric: acmc_metrics[metric] - d0ft_metrics[metric] for metric in METRICS
        },
        "static_audit": str(static_audit),
    }


def run_faruq_v3_acmc_paired_confirmation(
    data_root: str | Path,
    grouped_summary: str | Path,
    seed42_control: str | Path,
    output_root: str | Path,
    *,
    seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    """Confirm ACMC with two new optimization-matched pairs plus seed 42."""
    frozen_seeds = tuple(int(seed) for seed in seeds)
    if frozen_seeds != CONFIRMATION_SEEDS:
        raise ValueError(f"Konfirmasi baru dikunci pada seed {CONFIRMATION_SEEDS}")
    if not authorize_training:
        raise RuntimeError("Konfirmasi paired ACMC belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Faruq-v3 development tidak boleh memiliki split test")

    reports_root = output_root / "val_reports"
    base_root = output_root / "D0_base"
    d0ft_root = output_root / "D0FT"
    acmc_root = output_root / "ACMC1"
    static_root = output_root / "static_audits"
    reports_root.mkdir(parents=True, exist_ok=True)
    static_root.mkdir(parents=True, exist_ok=True)

    seed42 = _validate_seed42_control(seed42_control)
    per_seed: dict[str, dict] = {
        str(SEED42): {
            "source": str(Path(seed42_control).expanduser().resolve()),
            "results": seed42["results"],
            "control_deltas_d0ft_vs_d0": seed42["control_deltas_d0ft_vs_d0"],
            "head_deltas_acmc1_vs_d0ft": seed42["head_deltas_acmc1_vs_d0ft"],
        }
    }

    for seed in frozen_seeds:
        print(f"\n=== PAIRED ACMC CONFIRMATION | SEED {seed} ===", flush=True)
        baseline = run_faruq_v3_baseline(
            data_root, grouped_summary, base_root, seed=seed, device=device
        )
        d0_checkpoint = base_root / f"D0_seed{seed}" / "weights" / "best.pt"
        static_path = static_root / f"D0_seed{seed}_acmc_static.json"
        static = static_ambiguity_multilevel_audit(
            MODEL_YAML, d0_checkpoint, static_path, nc=21, image_size=128
        )
        if static["decision"] != "PASS":
            raise RuntimeError(f"Static audit ACMC seed {seed} gagal: {static_path}")

        d0ft_report, _ = run_d0ft_continuation(
            data_root, d0ft_root, d0_checkpoint, seed=seed, device=device
        )
        acmc = run_faruq_v3_acmc(
            data_root,
            grouped_summary,
            baseline["summary"],
            d0_checkpoint,
            static_path,
            acmc_root,
            seed=seed,
            device=device,
            authorize_training=True,
            confirmation_seed=True,
        )
        record = _seed_record(baseline, d0ft_report, acmc, static_path)
        record.update(
            {
                "d0_summary": baseline["summary"],
                "d0ft_report": str(d0ft_root / "val_reports" / f"D0FT_seed{seed}_val.json"),
                "acmc_summary": acmc["summary"],
            }
        )
        per_seed[str(seed)] = record

    all_seeds = (SEED42, *frozen_seeds)
    aggregate: dict[str, dict[str, float | int]] = {}
    for metric in METRICS:
        head_deltas = [
            float(per_seed[str(seed)]["head_deltas_acmc1_vs_d0ft"][metric])
            for seed in all_seeds
        ]
        aggregate[metric] = {
            "d0_mean": sum(float(per_seed[str(seed)]["results"]["D0"][metric]) for seed in all_seeds)
            / len(all_seeds),
            "d0ft_mean": sum(float(per_seed[str(seed)]["results"]["D0FT"][metric]) for seed in all_seeds)
            / len(all_seeds),
            "acmc1_mean": sum(float(per_seed[str(seed)]["results"]["ACMC1"][metric]) for seed in all_seeds)
            / len(all_seeds),
            "head_delta_mean": sum(head_deltas) / len(head_deltas),
            "head_delta_min": min(head_deltas),
            "head_improved_seeds": sum(delta > 0.0 for delta in head_deltas),
        }
    criteria = {
        "macro_head_gain_at_least_0_5_point": aggregate["macro_map50_95"]["head_delta_mean"] >= 0.005,
        "macro_head_improved_at_least_2_of_3": aggregate["macro_map50_95"]["head_improved_seeds"] >= 2,
        "bottom3_head_mean_not_lower": aggregate["bottom3_class_map50_95"]["head_delta_mean"] >= 0.0,
        "bottom3_head_improved_at_least_2_of_3": aggregate["bottom3_class_map50_95"]["head_improved_seeds"] >= 2,
        "worst_head_mean_drop_no_more_than_1_point": aggregate["worst_class_map50_95"]["head_delta_mean"] >= -0.01,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    payload = {
        "protocol": "faruq-v3-acmc-paired-optimization-confirmation-v1",
        "seeds": list(all_seeds),
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": decision,
        "next_action": (
            "AUTHORIZE_SINGLE_LOCKED_TEST_EVALUATION" if decision == "PASS"
            else "STOP_ACMC1_WITHOUT_TEST_OR_EXTRA_SEEDS"
        ),
    }
    summary = reports_root / "acmc1_paired_optimization_confirmation.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 paired ACMC optimization confirmation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--seed42-control", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CONFIRMATION_SEEDS))
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_acmc_paired_confirmation(
        args.data_root,
        args.grouped_summary,
        args.seed42_control,
        args.output_root,
        seeds=tuple(args.seeds),
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
