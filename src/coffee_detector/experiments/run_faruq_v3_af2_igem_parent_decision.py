"""Frozen three-seed decision for AF2FS + IGEM parent-preserving residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = (42, 123, 2026)
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
PROTOCOL = "faruq-v3-af2fs-igem-parent-confirmation-v1"

# Frozen in docs/FARUQ_V3_AF2FS_IGEM_PARENT_CONFIRMATION_PROTOCOL_2026-08-24.md
GATE = {
    "parent_macro_mean_delta_min": -0.002,
    "parent_bottom3_mean_delta_min": -0.010,
    "parent_worst_mean_delta_min": -0.010,
    "superiority_macro_mean_gain_min": 0.002,
    "superiority_macro_improved_seeds_min": 2,
    "superiority_bottom3_mean_delta_min": -0.005,
    "superiority_worst_mean_delta_min": -0.010,
    "tail_macro_mean_delta_min": -0.001,
    "tail_bottom3_mean_gain_min": 0.005,
    "tail_bottom3_improved_seeds_min": 2,
    "tail_worst_mean_gain_min": 0.010,
    "tail_worst_improved_seeds_min": 2,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def _normalize_parent(paths) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in paths:
        source = Path(path).expanduser().resolve()
        payload = _read(source)
        if payload.get("format") != "coffee_detector.af2_ffa.from_start_arm_result.v1" or payload.get("arm") != "AF2FS":
            raise RuntimeError("Parent harus result AF2FS from-start resmi")
        seed = int(payload.get("seed", -1))
        if seed not in SEEDS or seed in result:
            raise RuntimeError(f"Parent seed invalid/duplikat: {seed}")
        if payload.get("test_images_accessed") is not False:
            raise RuntimeError("Parent AF2FS harus test-locked")
        if len(payload.get("metrics", {}).get("map50_95_by_class", {})) != 21:
            raise RuntimeError(f"Parent AF2FS seed {seed} tidak memuat 21 kelas")
        payload = dict(payload)
        payload["_result_sha256"] = _sha256(source)
        result[seed] = payload
    if tuple(sorted(result)) != SEEDS:
        raise RuntimeError(f"Parent harus tepat seeds {SEEDS}")
    return result


def _normalize_arm(paths, expected_arm: str, conditioning: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = _read(path)
        if payload.get("format") != "coffee_detector.af2_parent_residual.igem_arm_result.v2":
            raise RuntimeError("Format result IGEM audited salah")
        if payload.get("protocol") != PROTOCOL:
            raise RuntimeError("Result berasal dari protokol lain")
        if (
            payload.get("arm") != expected_arm
            or payload.get("family") != "igem"
            or payload.get("conditioning") != conditioning
        ):
            raise RuntimeError(f"Result tidak cocok dengan arm {expected_arm}")
        if payload.get("parent_frozen") is not True or payload.get("trainable_scope") != "igem_residual_only":
            raise RuntimeError(f"{expected_arm} tidak membuktikan residual-only training")
        if payload.get("evaluation_split") != "val" or payload.get("test_images_accessed") is not False:
            raise RuntimeError(f"{expected_arm} harus validation-only dan test terkunci")
        if payload.get("baseline_source") != "canonical_AF2FS_parent_result":
            raise RuntimeError(f"{expected_arm} tidak memakai baseline parent canonical")
        seed = int(payload.get("seed", -1))
        if seed not in SEEDS or seed in result:
            raise RuntimeError(f"Seed invalid/duplikat: {seed}")
        if len(payload.get("metrics", {}).get("map50_95_by_class", {})) != 21:
            raise RuntimeError(f"{expected_arm} seed {seed} tidak memuat 21 kelas")
        if len(payload.get("baseline_metrics", {}).get("map50_95_by_class", {})) != 21:
            raise RuntimeError(f"Parent baseline {expected_arm} seed {seed} tidak memuat 21 kelas")
        result[seed] = payload
    if tuple(sorted(result)) != SEEDS:
        raise RuntimeError(f"Harus tepat seeds {SEEDS}")
    return result


def _metric_row(parent, control, candidate, metric: str) -> dict[str, Any]:
    parent_values = [float(parent[s]["metrics"][metric]) for s in SEEDS]
    control_values = [float(control[s]["metrics"][metric]) for s in SEEDS]
    candidate_values = [float(candidate[s]["metrics"][metric]) for s in SEEDS]
    vs_control = [right - left for left, right in zip(control_values, candidate_values)]
    vs_parent = [right - left for left, right in zip(parent_values, candidate_values)]
    return {
        "parent_mean": float(np.mean(parent_values)),
        "control_mean": float(np.mean(control_values)),
        "candidate_mean": float(np.mean(candidate_values)),
        "parent_sd": float(np.std(parent_values, ddof=1)),
        "control_sd": float(np.std(control_values, ddof=1)),
        "candidate_sd": float(np.std(candidate_values, ddof=1)),
        "candidate_minus_control_mean": float(np.mean(vs_control)),
        "candidate_minus_parent_mean": float(np.mean(vs_parent)),
        "candidate_minus_control_improved_seeds": int(sum(delta > 0.0 for delta in vs_control)),
        "candidate_minus_parent_improved_seeds": int(sum(delta > 0.0 for delta in vs_parent)),
        "candidate_minus_control": {str(seed): float(delta) for seed, delta in zip(SEEDS, vs_control)},
        "candidate_minus_parent": {str(seed): float(delta) for seed, delta in zip(SEEDS, vs_parent)},
        "parent_values": {str(seed): float(value) for seed, value in zip(SEEDS, parent_values)},
        "control_values": {str(seed): float(value) for seed, value in zip(SEEDS, control_values)},
        "candidate_values": {str(seed): float(value) for seed, value in zip(SEEDS, candidate_values)},
    }


def run_igem_parent_decision(parent_paths, control_paths, candidate_paths, output: str | Path) -> dict[str, Any]:
    parent = _normalize_parent(parent_paths)
    control = _normalize_arm(control_paths, "AF2IGEM0", "zero")
    candidate = _normalize_arm(candidate_paths, "AF2IGEM1", "feature")

    for seed in SEEDS:
        canonical_checkpoint_sha = parent[seed]["checkpoint_sha256"]
        canonical_result_sha = parent[seed]["_result_sha256"]
        for label, payload in (("control", control[seed]), ("candidate", candidate[seed])):
            if payload.get("initial_af2_checkpoint_sha256") != canonical_checkpoint_sha:
                raise RuntimeError(f"Seed {seed}: {label} tidak memakai AF2FS checkpoint canonical")
            if payload.get("parent_result_sha256") != canonical_result_sha:
                raise RuntimeError(f"Seed {seed}: {label} tidak terikat ke AF2FS result canonical")
            if payload.get("baseline_metrics") != parent[seed]["metrics"]:
                raise RuntimeError(f"Seed {seed}: baseline {label} bukan salinan metric AF2FS canonical")
        if control[seed].get("run_contract_sha256") is None or candidate[seed].get("run_contract_sha256") is None:
            raise RuntimeError(f"Seed {seed}: run contract tidak tercatat")

    aggregate = {metric: _metric_row(parent, control, candidate, metric) for metric in METRICS}
    macro = aggregate["macro_map50_95"]
    bottom3 = aggregate["bottom3_class_map50_95"]
    worst = aggregate["worst_class_map50_95"]

    parent_safety = {
        "macro_mean_not_below_parent_by_more_than_0_2pp":
            macro["candidate_minus_parent_mean"] >= GATE["parent_macro_mean_delta_min"],
        "bottom3_mean_not_below_parent_by_more_than_1pp":
            bottom3["candidate_minus_parent_mean"] >= GATE["parent_bottom3_mean_delta_min"],
        "worst_mean_not_below_parent_by_more_than_1pp":
            worst["candidate_minus_parent_mean"] >= GATE["parent_worst_mean_delta_min"],
    }
    superiority = {
        "macro_mean_gain_at_least_0_2pp":
            macro["candidate_minus_control_mean"] >= GATE["superiority_macro_mean_gain_min"],
        "macro_improves_at_least_2_of_3":
            macro["candidate_minus_control_improved_seeds"] >= GATE["superiority_macro_improved_seeds_min"],
        "bottom3_mean_loss_at_most_0_5pp":
            bottom3["candidate_minus_control_mean"] >= GATE["superiority_bottom3_mean_delta_min"],
        "worst_mean_loss_at_most_1pp":
            worst["candidate_minus_control_mean"] >= GATE["superiority_worst_mean_delta_min"],
    }
    tail_pareto = {
        "macro_mean_loss_at_most_0_1pp":
            macro["candidate_minus_control_mean"] >= GATE["tail_macro_mean_delta_min"],
        "bottom3_mean_gain_at_least_0_5pp":
            bottom3["candidate_minus_control_mean"] >= GATE["tail_bottom3_mean_gain_min"],
        "bottom3_improves_at_least_2_of_3":
            bottom3["candidate_minus_control_improved_seeds"] >= GATE["tail_bottom3_improved_seeds_min"],
        "worst_mean_gain_at_least_1pp":
            worst["candidate_minus_control_mean"] >= GATE["tail_worst_mean_gain_min"],
        "worst_improves_at_least_2_of_3":
            worst["candidate_minus_control_improved_seeds"] >= GATE["tail_worst_improved_seeds_min"],
    }

    safety_pass = all(parent_safety.values())
    superiority_pass = all(superiority.values())
    tail_pass = all(tail_pareto.values())
    passed = safety_pass and (superiority_pass or tail_pass)
    result = {
        "format": "coffee_detector.af2_parent_residual.igem_three_seed_decision.v2",
        "comparison": "AF2IGEM1_vs_AF2IGEM0_with_canonical_seed_matched_frozen_AF2FS_parents",
        "seeds": list(SEEDS),
        "gate": GATE,
        "aggregate": aggregate,
        "criteria": {
            "parent_safety": parent_safety,
            "superiority_route": superiority,
            "tail_pareto_route": tail_pareto,
            "parent_safety_pass": safety_pass,
            "superiority_route_pass": superiority_pass,
            "tail_pareto_route_pass": tail_pass,
            "all_21_classes_present": True,
            "canonical_parent_binding_verified": True,
            "test_not_opened": True,
        },
        "decision": "RETAIN" if passed else "REJECT",
        "next": "PROCEED_TO_SEPARATE_SAF_HYPOTHESIS" if passed else "STOP_AF2FS_IGEM_RESIDUAL_ROUTE",
        "claim_if_retain": (
            "Under matched three-seed development-validation training with canonical AF2FS parents frozen, "
            "real P3/P4/P5 conditioning gives the IGEM residual useful classification information beyond "
            "its zero-information capacity/optimization control."
        ),
        "interpretation_boundary": (
            "Development-validation evidence only. The locked test stays closed and SAF/STB are separate hypotheses."
        ),
        "test_opened": False,
    }
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide AF2FS + IGEM frozen-parent paired confirmation")
    parser.add_argument("--parent", nargs=3, required=True)
    parser.add_argument("--control", nargs=3, required=True)
    parser.add_argument("--candidate", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_igem_parent_decision(args.parent, args.control, args.candidate, args.output)


if __name__ == "__main__":
    main()
