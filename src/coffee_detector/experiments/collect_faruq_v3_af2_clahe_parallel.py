"""Collect three completed parallel CLAHE workers into the frozen comparison.

No training occurs here. The collector requires exactly the frozen seeds 42/123/2026,
reads their CLAHE validation reports plus the existing frozen AF2/D0FT confirmation,
and applies the same decision functions as the canonical sequential runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.experiments.run_faruq_v3_af2_clahe_control import (
    METRICS,
    SEEDS,
    _af2_specific_decision,
    _clahe_superiority_decision,
    _generic_enhancement_decision,
    _metrics,
    _paired_summary,
    _validate_reference,
)


def collect(
    af2_confirmation: str | Path,
    output_root: str | Path,
) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    reports = output_root / "val_reports"
    reference = _validate_reference(af2_confirmation)

    per_seed: dict[str, dict] = {}
    for seed in SEEDS:
        worker_path = reports / f"CLAHE_LAB_seed{seed}_worker.json"
        val_path = reports / f"CLAHE_LAB_seed{seed}_val.json"
        if not worker_path.is_file() or not val_path.is_file():
            raise FileNotFoundError(
                f"Seed {seed} belum lengkap. Butuh {worker_path.name} dan {val_path.name}"
            )
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        if (
            worker.get("protocol")
            != "faruq-v3-af2-vs-clahe-classical-enhancement-control-v1"
            or int(worker.get("seed", -1)) != seed
            or worker.get("evaluation_split") != "val"
            or worker.get("test_images_accessed") is not False
            or worker.get("test_opened") is not False
        ):
            raise RuntimeError(f"Worker seed {seed} tidak kompatibel")
        val = json.loads(val_path.read_text(encoding="utf-8"))
        frozen = reference["per_seed"][str(seed)]
        per_seed[str(seed)] = {
            "D0FT": _metrics(frozen["D0FT"]),
            "AF2": _metrics(frozen["AF2"]),
            "CLAHE_LAB": _metrics(val),
        }

    clahe_vs_d0ft = _paired_summary(per_seed, "D0FT", "CLAHE_LAB")
    af2_vs_clahe = _paired_summary(per_seed, "CLAHE_LAB", "AF2")
    clahe_vs_af2 = _paired_summary(per_seed, "AF2", "CLAHE_LAB")

    generic_criteria, generic_decision = _generic_enhancement_decision(clahe_vs_d0ft)
    af2_criteria, af2_decision = _af2_specific_decision(af2_vs_clahe)
    clahe_criteria, clahe_decision = _clahe_superiority_decision(clahe_vs_af2)

    if af2_decision == "PASS":
        interpretation = "AF2_SPECIFIC_ADVANTAGE_SUPPORTED"
    elif clahe_decision == "PASS":
        interpretation = "CLAHE_SUPERIOR_UNDER_FROZEN_CONTROL"
    else:
        interpretation = "NO_DIRECTIONAL_SUPERIORITY_ESTABLISHED"

    result = {
        "protocol": "faruq-v3-af2-vs-clahe-classical-enhancement-control-v1",
        "execution_mode": "parallel_workers_collected",
        "seeds": list(SEEDS),
        "models": ["D0FT", "CLAHE_LAB", "AF2"],
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "per_seed": per_seed,
        "comparisons": {
            "CLAHE_minus_D0FT": clahe_vs_d0ft,
            "AF2_minus_CLAHE": af2_vs_clahe,
            "CLAHE_minus_AF2": clahe_vs_af2,
        },
        "decisions": {
            "generic_clahe_effect": {
                "decision": generic_decision,
                "criteria": generic_criteria,
            },
            "af2_beyond_clahe": {
                "decision": af2_decision,
                "criteria": af2_criteria,
            },
            "clahe_superior_to_af2": {
                "decision": clahe_decision,
                "criteria": clahe_criteria,
            },
        },
        "interpretation": interpretation,
        "metrics": list(METRICS),
    }
    summary = reports / "af2_vs_clahe_classical_enhancement_control.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect parallel AF2-vs-CLAHE workers")
    parser.add_argument("--af2-confirmation", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.af2_confirmation, args.output_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
