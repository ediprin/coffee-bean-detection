from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("AF2BASE", "AF2SPDS", "AF2CUE1", "AF2DECAY1")
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _result_path(original_root: Path, refinement_root: Path, arm: str) -> Path:
    root = original_root if arm in {"AF2BASE", "AF2SPDS"} else refinement_root
    return root / "val_reports" / f"{arm}_seed42_result.json"


def _load_result(original_root: Path, refinement_root: Path, arm: str) -> dict[str, Any]:
    path = _result_path(original_root, refinement_root, arm)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("arm") != arm or int(payload.get("seed", -1)) != 42:
        raise RuntimeError(f"Kontrak arm/seed tidak cocok: {path}")
    if payload.get("test_images_accessed") is not False:
        raise RuntimeError(f"Test lock gagal: {arm}")
    metrics = payload.get("metrics", {})
    classes = metrics.get("map50_95_by_class")
    if not isinstance(classes, dict) or len(classes) != 21:
        raise RuntimeError(f"AP 21 kelas tidak lengkap: {arm}")
    if metrics.get("classes_without_ground_truth"):
        raise RuntimeError(f"Validation kehilangan kelas: {arm}")
    return payload


def _headline(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values, axis=-1)
    return np.stack(
        (values.mean(axis=-1), ordered[..., :3].mean(axis=-1), ordered[..., 0]),
        axis=-1,
    )


def paired_class_bootstrap(
    comparator: dict[str, float],
    candidate: dict[str, float],
    *,
    iterations: int = 10_000,
    seed: int = 20260829,
) -> dict[str, Any]:
    """Exploratory paired bootstrap over the 21 class identities.

    This estimates sensitivity to class composition. It is deliberately not
    labelled an image-, parent-, or test-level confidence interval because AP
    observations for classes share the same validation images.
    """

    names = sorted(comparator)
    if names != sorted(candidate) or len(names) != 21:
        raise ValueError("Bootstrap membutuhkan pasangan 21 kelas yang identik")
    if iterations <= 0:
        raise ValueError("iterations harus positif")
    left = np.asarray([float(comparator[name]) for name in names], dtype=np.float64)
    right = np.asarray([float(candidate[name]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(names), size=(iterations, len(names)))
    deltas = _headline(right[indices]) - _headline(left[indices])
    point = _headline(right[None, :])[0] - _headline(left[None, :])[0]
    result: dict[str, Any] = {
        "iterations": iterations,
        "seed": seed,
        "unit": "validation_class_identity",
        "independence_warning": (
            "Exploratory class-composition bootstrap only; classes share images "
            "and this is not a parent/image-level inferential test."
        ),
        "metrics": {},
    }
    for index, metric in enumerate(METRICS):
        values = deltas[:, index]
        result["metrics"][metric] = {
            "point_delta": float(point[index]),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "probability_positive": float(np.mean(values > 0.0)),
            "probability_nonnegative": float(np.mean(values >= 0.0)),
        }
    return result


def run_af2_spds_refinement_posthoc(
    original_root: str | Path,
    refinement_root: str | Path,
    output: str | Path,
    *,
    iterations: int = 10_000,
    seed: int = 20260829,
    print_json: bool = True,
) -> dict[str, Any]:
    original_root = Path(original_root).expanduser().resolve()
    refinement_root = Path(refinement_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    payloads = {
        arm: _load_result(original_root, refinement_root, arm) for arm in ARMS
    }
    class_ap = {
        arm: {
            str(name): float(value)
            for name, value in payload["metrics"]["map50_95_by_class"].items()
        }
        for arm, payload in payloads.items()
    }
    names = sorted(class_ap["AF2BASE"])
    if any(sorted(class_ap[arm]) != names for arm in ARMS):
        raise RuntimeError("Ontology kelas antar-arm tidak identik")

    per_class = []
    for name in names:
        row: dict[str, Any] = {"class_name": name}
        row.update({arm: class_ap[arm][name] for arm in ARMS})
        row["cue1_minus_base"] = row["AF2CUE1"] - row["AF2BASE"]
        row["cue1_minus_spds"] = row["AF2CUE1"] - row["AF2SPDS"]
        row["decay1_minus_spds"] = row["AF2DECAY1"] - row["AF2SPDS"]
        per_class.append(row)

    cue1_vs_base = paired_class_bootstrap(
        class_ap["AF2BASE"], class_ap["AF2CUE1"], iterations=iterations, seed=seed
    )
    cue1_vs_spds = paired_class_bootstrap(
        class_ap["AF2SPDS"],
        class_ap["AF2CUE1"],
        iterations=iterations,
        seed=seed + 1,
    )
    cue1_base_delta = cue1_vs_base["metrics"]
    exploratory_retain = (
        all(cue1_base_delta[metric]["point_delta"] > 0.0 for metric in METRICS)
        and cue1_vs_spds["metrics"]["macro_map50_95"]["point_delta"] > 0.0
        and cue1_vs_spds["metrics"]["worst_class_map50_95"]["point_delta"] >= 0.0
    )
    by_base = sorted(per_class, key=lambda row: (row["cue1_minus_base"], row["class_name"]))
    by_spds = sorted(per_class, key=lambda row: (row["cue1_minus_spds"], row["class_name"]))
    result = {
        "format": "coffee_detector.af2_spds_refinement.posthoc.v1",
        "scope": "validation_only_existing_reports_no_training",
        "formal_frozen_decision": "FAIL_KILL_GATE",
        "formal_next": "RETAIN_ORIGINAL_AF2",
        "exploratory_research_status": (
            "RETAIN_PARETO_EXPLORATORY" if exploratory_retain else "DO_NOT_RETAIN"
        ),
        "per_class": per_class,
        "class_summary": {
            "cue1_improved_vs_base": sum(row["cue1_minus_base"] > 0 for row in per_class),
            "cue1_declined_vs_base": sum(row["cue1_minus_base"] < 0 for row in per_class),
            "cue1_improved_vs_spds": sum(row["cue1_minus_spds"] > 0 for row in per_class),
            "cue1_declined_vs_spds": sum(row["cue1_minus_spds"] < 0 for row in per_class),
            "largest_gains_vs_base": list(reversed(by_base[-7:])),
            "largest_losses_vs_base": by_base[:7],
            "largest_gains_vs_spds": list(reversed(by_spds[-7:])),
            "largest_losses_vs_spds": by_spds[:7],
        },
        "paired_class_bootstrap": {
            "AF2CUE1_vs_AF2BASE": cue1_vs_base,
            "AF2CUE1_vs_AF2SPDS": cue1_vs_spds,
        },
        "training_executed": False,
        "test_opened": False,
        "claim_boundary": (
            "Post-hoc exploratory evidence; does not override the frozen gate or "
            "authorize a confirmatory superiority claim."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if print_json:
        print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc per-class and paired-class bootstrap for AF2CUE1"
    )
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--refinement-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--quiet", action="store_true", help="Simpan JSON tanpa mencetak payload penuh"
    )
    args = parser.parse_args()
    run_af2_spds_refinement_posthoc(
        args.original_root,
        args.refinement_root,
        args.output,
        iterations=args.iterations,
        seed=args.seed,
        print_json=not args.quiet,
    )


if __name__ == "__main__":
    main()
