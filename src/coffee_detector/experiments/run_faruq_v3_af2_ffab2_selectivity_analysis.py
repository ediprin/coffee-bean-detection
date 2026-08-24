"""Validation-only diagnosis of why FFAB2 tail gains do not lift Macro.

This analysis reuses completed AF2FS/AF2FFAB2FS checkpoints. It performs no
training and never opens test. The search is explicitly exploratory: any
selected runtime setting must be retrained and re-confirmed before a thesis
upgrade claim is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from coffee_detector.af2_ffa import AF2FFADetectHead
from coffee_detector.dataset import discover_layout
from coffee_detector.evaluate import _classwise_summary

SEEDS = (42, 123, 2026)
BETA_VALUES = (0.0, 0.25, 0.50, 0.75, 1.0)
LEVEL_SUBSETS = (
    ("P3",),
    ("P4",),
    ("P5",),
    ("P3", "P4"),
    ("P3", "P5"),
    ("P4", "P5"),
    ("P3", "P4", "P5"),
)
PARENT_MIX_VALUES = (0.25, 0.50, 0.75, 1.0)
AMBIGUITY_MARGINS = (0.05, 0.10, 0.15, 0.20)
AMBIGUITY_TEMPERATURE = 0.05

# Frozen before executing this follow-up analysis. This is an exploratory
# authorization gate, not a confirmation claim.
DIAGNOSTIC_GATE = {
    "macro_mean_gain_min": 0.0025,       # +0.25 pp
    "macro_improved_seeds_min": 2,
    "bottom3_mean_gain_min": 0.0050,     # +0.50 pp
    "bottom3_improved_seeds_min": 2,
    "worst_mean_delta_min": 0.0,
    "worst_improved_seeds_min": 2,
}


def _read(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_results(paths: Iterable[str | Path], arm: str) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = _read(path)
        if payload.get("format") != "coffee_detector.af2_ffa.from_start_arm_result.v1":
            raise RuntimeError(f"Format result tidak valid: {path}")
        if payload.get("arm") != arm:
            raise RuntimeError(f"Arm result bukan {arm}: {path}")
        seed = int(payload.get("seed", -1))
        if seed not in SEEDS or seed in results:
            raise RuntimeError(f"Seed result invalid/duplikat: {seed}")
        if payload.get("evaluation_split") != "val" or payload.get("test_images_accessed") is not False:
            raise RuntimeError("Analisis hanya menerima result validation dengan test terkunci")
        metrics = payload.get("metrics") or {}
        required = {
            "macro_map50_95",
            "bottom3_class_map50_95",
            "worst_class_map50_95",
            "map50_95_by_class",
        }
        if not required.issubset(metrics):
            raise RuntimeError(f"Result seed {seed} tidak memiliki metrik classwise lengkap")
        results[seed] = payload
    if tuple(sorted(results)) != SEEDS:
        raise RuntimeError(f"Harus ada tepat seed {SEEDS}; ditemukan {sorted(results)}")
    return results


def _pair_guard(af2: dict[int, dict], ffab2: dict[int, dict]) -> None:
    for seed in SEEDS:
        left = af2[seed]
        right = ffab2[seed]
        if left.get("initial_d0_checkpoint_sha256") != right.get("initial_d0_checkpoint_sha256"):
            raise RuntimeError(f"Seed {seed}: AF2FS/FFAB2FS tidak berasal dari D0 yang sama")


def _class_delta_analysis(af2: dict[int, dict], ffab2: dict[int, dict]) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    class_values: dict[str, list[float]] = {}
    class_signs: dict[str, list[bool]] = {}
    for seed in SEEDS:
        base = af2[seed]["metrics"]["map50_95_by_class"]
        refined = ffab2[seed]["metrics"]["map50_95_by_class"]
        if set(base) != set(refined):
            raise RuntimeError(f"Seed {seed}: kelas AF2FS dan FFAB2FS tidak identik")
        deltas = {name: float(refined[name]) - float(base[name]) for name in sorted(base)}
        by_seed[str(seed)] = {
            "deltas": deltas,
            "improved_classes": sum(value > 0 for value in deltas.values()),
            "regressed_classes": sum(value < 0 for value in deltas.values()),
        }
        for name, value in deltas.items():
            class_values.setdefault(name, []).append(value)
            class_signs.setdefault(name, []).append(value > 0)
    aggregate = []
    for name in sorted(class_values):
        values = class_values[name]
        aggregate.append(
            {
                "class": name,
                "mean_delta": float(np.mean(values)),
                "min_delta": float(np.min(values)),
                "max_delta": float(np.max(values)),
                "improved_seeds": int(sum(class_signs[name])),
                "deltas": {str(seed): float(value) for seed, value in zip(SEEDS, values)},
            }
        )
    aggregate.sort(key=lambda row: row["mean_delta"], reverse=True)
    return {
        "by_seed": by_seed,
        "aggregate": aggregate,
        "consistent_help": [row for row in aggregate if row["improved_seeds"] == 3],
        "consistent_harm": [row for row in aggregate if row["improved_seeds"] == 0],
    }


def _variant_specs() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for beta in BETA_VALUES:
        variants.append(
            {
                "id": f"beta_{beta:.2f}",
                "family": "strength",
                "strength": beta,
                "active_levels": ["P3", "P4", "P5"],
                "fusion_mode": "replace",
            }
        )
    for levels in LEVEL_SUBSETS:
        variants.append(
            {
                "id": "levels_" + "_".join(levels),
                "family": "levels",
                "strength": 1.0,
                "active_levels": list(levels),
                "fusion_mode": "replace",
            }
        )
    for mix in PARENT_MIX_VALUES:
        variants.append(
            {
                "id": f"parent_mix_{mix:.2f}",
                "family": "parent_residual",
                "strength": 1.0,
                "active_levels": ["P3", "P4", "P5"],
                "fusion_mode": "parent_residual",
                "residual_mix": mix,
                "ambiguity_gate": "none",
            }
        )
    for margin in AMBIGUITY_MARGINS:
        variants.append(
            {
                "id": f"ambiguity_margin_{margin:.2f}",
                "family": "ambiguity",
                "strength": 1.0,
                "active_levels": ["P3", "P4", "P5"],
                "fusion_mode": "parent_residual",
                "residual_mix": 1.0,
                "ambiguity_gate": "margin",
                "ambiguity_margin": margin,
                "ambiguity_temperature": AMBIGUITY_TEMPERATURE,
            }
        )
    # Deduplicate exact all-level replace beta=1 variants.
    unique: dict[str, dict[str, Any]] = {}
    fingerprints: set[str] = set()
    for variant in variants:
        semantic = {key: value for key, value in variant.items() if key not in {"id", "family"}}
        fingerprint = json.dumps(semantic, sort_keys=True)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        unique[variant["id"]] = variant
    return list(unique.values())


def _index_checkpoints(roots: Iterable[str | Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else list(root.rglob("best.pt"))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            digest = _sha256(candidate)
            index.setdefault(digest, candidate)
    return index


def _resolve_checkpoint(result: dict[str, Any], index: dict[str, Path]) -> Path:
    expected = str(result.get("checkpoint_sha256", ""))
    direct = Path(str(result.get("checkpoint", ""))).expanduser()
    if direct.is_file() and _sha256(direct.resolve()) == expected:
        return direct.resolve()
    if expected in index:
        return index[expected]
    raise FileNotFoundError(f"Checkpoint SHA {expected} tidak ditemukan di checkpoint roots")


def _runtime_metrics(yolo, data_root: Path, device: str, variant: dict[str, Any]) -> dict[str, Any]:
    head = yolo.model.model[-1]
    if not isinstance(head, AF2FFADetectHead):
        raise TypeError(f"Checkpoint bukan AF2FFADetectHead: {type(head).__name__}")
    head.set_runtime_ablation(
        strength=float(variant["strength"]),
        active_levels=variant["active_levels"],
        fusion_mode=variant.get("fusion_mode"),
        residual_mix=variant.get("residual_mix"),
        ambiguity_gate=variant.get("ambiguity_gate"),
        ambiguity_margin=variant.get("ambiguity_margin"),
        ambiguity_temperature=variant.get("ambiguity_temperature"),
    )
    layout = discover_layout(data_root)
    metrics = yolo.val(
        data=str(layout.yaml_path),
        split="val",
        device=device,
        plots=False,
        verbose=False,
    )
    summary = _classwise_summary(metrics.box, layout.names)
    if summary.get("classes_without_ground_truth"):
        raise RuntimeError("Validation runtime kehilangan kelas")
    return summary


def _aggregate_variant(per_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"per_seed": {str(seed): per_seed[seed] for seed in SEEDS}}
    for metric in ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95"):
        values = [float(per_seed[seed][metric]) for seed in SEEDS]
        result[metric] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "values": {str(seed): value for seed, value in zip(SEEDS, values)},
        }
    return result


def _baseline_aggregate(af2: dict[int, dict]) -> dict[str, Any]:
    per_seed = {seed: af2[seed]["metrics"] for seed in SEEDS}
    return _aggregate_variant(per_seed)


def _score_against_af2(variant: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "macro_map50_95": "macro",
        "bottom3_class_map50_95": "bottom3",
        "worst_class_map50_95": "worst",
    }
    comparison: dict[str, Any] = {}
    for metric, short in mapping.items():
        deltas = {
            str(seed): float(variant[metric]["values"][str(seed)])
            - float(baseline[metric]["values"][str(seed)])
            for seed in SEEDS
        }
        comparison[short] = {
            "delta_mean": float(np.mean(list(deltas.values()))),
            "improved_seeds": int(sum(value > 0 for value in deltas.values())),
            "deltas": deltas,
        }
    gate = {
        "macro_mean": comparison["macro"]["delta_mean"] >= DIAGNOSTIC_GATE["macro_mean_gain_min"],
        "macro_seeds": comparison["macro"]["improved_seeds"] >= DIAGNOSTIC_GATE["macro_improved_seeds_min"],
        "bottom3_mean": comparison["bottom3"]["delta_mean"] >= DIAGNOSTIC_GATE["bottom3_mean_gain_min"],
        "bottom3_seeds": comparison["bottom3"]["improved_seeds"] >= DIAGNOSTIC_GATE["bottom3_improved_seeds_min"],
        "worst_mean": comparison["worst"]["delta_mean"] >= DIAGNOSTIC_GATE["worst_mean_delta_min"],
        "worst_seeds": comparison["worst"]["improved_seeds"] >= DIAGNOSTIC_GATE["worst_improved_seeds_min"],
    }
    comparison["gate"] = gate
    comparison["eligible_for_retrain_screen"] = all(gate.values())
    return comparison


def run_analysis(
    af2_result_paths: Iterable[str | Path],
    ffab2_result_paths: Iterable[str | Path],
    checkpoint_roots: Iterable[str | Path],
    data_root: str | Path,
    output: str | Path,
    *,
    device: str = "0",
    skip_runtime: bool = False,
) -> dict[str, Any]:
    data_root = Path(data_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    af2 = _normalize_results(af2_result_paths, "AF2FS")
    ffab2 = _normalize_results(ffab2_result_paths, "AF2FFAB2FS")
    _pair_guard(af2, ffab2)
    baseline = _baseline_aggregate(af2)
    payload: dict[str, Any] = {
        "format": "coffee_detector.af2_ffa.selectivity_analysis.v1",
        "status": "EXPLORATORY_VALIDATION_ONLY",
        "seeds": list(SEEDS),
        "test_opened": False,
        "diagnostic_gate": DIAGNOSTIC_GATE,
        "per_class": _class_delta_analysis(af2, ffab2),
        "af2fs_baseline": baseline,
        "runtime_variants": [],
        "decision": "ANALYSIS_ONLY",
        "training_authorized": False,
    }
    if not skip_runtime:
        from ultralytics import YOLO

        checkpoint_index = _index_checkpoints(checkpoint_roots)
        variants = _variant_specs()
        for variant in variants:
            per_seed: dict[int, dict[str, Any]] = {}
            for seed in SEEDS:
                checkpoint = _resolve_checkpoint(ffab2[seed], checkpoint_index)
                yolo = YOLO(str(checkpoint))
                per_seed[seed] = _runtime_metrics(yolo, data_root, device, variant)
            aggregate = _aggregate_variant(per_seed)
            comparison = _score_against_af2(aggregate, baseline)
            payload["runtime_variants"].append(
                {"variant": variant, "aggregate": aggregate, "vs_af2fs": comparison}
            )
        eligible = [
            row for row in payload["runtime_variants"]
            if row["vs_af2fs"]["eligible_for_retrain_screen"]
        ]
        eligible.sort(
            key=lambda row: (
                row["vs_af2fs"]["macro"]["delta_mean"],
                row["vs_af2fs"]["bottom3"]["delta_mean"],
                row["vs_af2fs"]["worst"]["delta_mean"],
            ),
            reverse=True,
        )
        if eligible:
            payload["decision"] = "DIAGNOSTIC_CANDIDATE_FOUND"
            payload["selected_candidate"] = eligible[0]["variant"]
            payload["selected_candidate_comparison"] = eligible[0]["vs_af2fs"]
            payload["training_authorized"] = True
            payload["claim_boundary"] = (
                "Validation-tuned runtime candidate only; retraining and independent confirmation required."
            )
        else:
            payload["decision"] = "NO_RUNTIME_CANDIDATE_PASSES_GATE"
            payload["next"] = "PARENT_PRESERVING_ARCHITECTURE_REMAINS_IMPLEMENTED_BUT_NOT_AUTHORIZED"
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"],
        "training_authorized": payload["training_authorized"],
        "selected_candidate": payload.get("selected_candidate"),
        "test_opened": False,
    }, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="AF2/FFAB2 selectivity diagnosis; validation only")
    parser.add_argument("--af2-result", action="append", required=True, dest="af2_results")
    parser.add_argument("--ffab2-result", action="append", required=True, dest="ffab2_results")
    parser.add_argument("--checkpoint-root", action="append", default=[], dest="checkpoint_roots")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-runtime", action="store_true")
    args = parser.parse_args()
    if len(args.af2_results) != 3 or len(args.ffab2_results) != 3:
        parser.error("Harus memberikan tepat tiga --af2-result dan tiga --ffab2-result")
    if not args.skip_runtime and not args.checkpoint_roots:
        parser.error("Runtime sweep memerlukan minimal satu --checkpoint-root")
    run_analysis(
        args.af2_results,
        args.ffab2_results,
        args.checkpoint_roots,
        args.data_root,
        args.output,
        device=args.device,
        skip_runtime=args.skip_runtime,
    )


if __name__ == "__main__":
    main()
