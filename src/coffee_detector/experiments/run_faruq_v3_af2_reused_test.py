from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from coffee_detector.experiments.run_faruq_v3_acmc import METRICS
from coffee_detector.experiments.run_faruq_v3_acmc_locked_test import (
    _evaluate_checkpoint,
    _load_json,
    _paired_parent_bootstrap,
    _sha256,
    _ultralytics_test_yaml,
)
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES


SEEDS = (42, 123, 2026)


def _validate_test_report(
    report: dict, *, manifest_hash: str, require_protocol: str | None = None
) -> None:
    if report.get("split") != "test":
        raise RuntimeError("Report bukan split test")
    if report.get("test_manifest_sha256") != manifest_hash:
        raise RuntimeError("Report memakai manifest test berbeda")
    if report.get("training_executed") is not False:
        raise RuntimeError("Report test tidak boleh menjalankan training")
    if report.get("test_images_accessed") is not True:
        raise RuntimeError("Report tidak merekam akses test")
    if not isinstance(report.get("prediction_observations"), list):
        raise RuntimeError("Report tidak memiliki observasi untuk bootstrap")
    if require_protocol is not None and report.get("protocol") != require_protocol:
        raise RuntimeError("Protocol report AF2 tidak sesuai")
    metrics = report.get("metrics", {})
    if any(metric not in metrics for metric in METRICS):
        raise RuntimeError("Report kehilangan metrik utama")


def build_af2_reused_test_summary(
    d0ft_reports: list[str | Path],
    af2_reports: list[str | Path],
    test_manifest: str | Path,
    output: str | Path,
    *,
    bootstrap_iterations: int = 1000,
) -> dict:
    if len(d0ft_reports) != 3 or len(af2_reports) != 3:
        raise ValueError("D0FT dan AF2 masing-masing memerlukan tiga report")
    manifest_hash = _sha256(test_manifest)
    reports = {}
    observations = {}
    for index, seed in enumerate(SEEDS):
        control = _load_json(d0ft_reports[index], f"D0FT seed {seed}")
        candidate = _load_json(af2_reports[index], f"AF2 seed {seed}")
        _validate_test_report(control, manifest_hash=manifest_hash)
        _validate_test_report(
            candidate,
            manifest_hash=manifest_hash,
            require_protocol="faruq-v3-af2-reused-test-posthoc-report-v1",
        )
        reports[seed] = {"D0FT": control, "AF2": candidate}
        observations[str(seed)] = {
            "D0FT": control["prediction_observations"],
            "AF2": candidate["prediction_observations"],
        }

    per_seed = {}
    for seed in SEEDS:
        values = {
            arm: {metric: float(reports[seed][arm]["metrics"][metric]) for metric in METRICS}
            for arm in ("D0FT", "AF2")
        }
        per_seed[str(seed)] = {
            "results": values,
            "deltas": {
                metric: values["AF2"][metric] - values["D0FT"][metric]
                for metric in METRICS
            },
        }

    aggregate = {}
    for metric in METRICS:
        controls = [per_seed[str(seed)]["results"]["D0FT"][metric] for seed in SEEDS]
        candidates = [per_seed[str(seed)]["results"]["AF2"][metric] for seed in SEEDS]
        deltas = [candidate - control for control, candidate in zip(controls, candidates)]
        aggregate[metric] = {
            "d0ft_mean": statistics.mean(controls),
            "d0ft_std": statistics.stdev(controls),
            "af2_mean": statistics.mean(candidates),
            "af2_std": statistics.stdev(candidates),
            "delta_mean": statistics.mean(deltas),
            "delta_std": statistics.stdev(deltas),
            "delta_min": min(deltas),
            "improved_seeds": sum(delta > 0 for delta in deltas),
        }

    classwise = {}
    for name in SNI21_CLASSES:
        controls = [
            float(reports[seed]["D0FT"]["metrics"]["map50_95_by_class"][name])
            for seed in SEEDS
        ]
        candidates = [
            float(reports[seed]["AF2"]["metrics"]["map50_95_by_class"][name])
            for seed in SEEDS
        ]
        classwise[name] = {
            "d0ft_mean": statistics.mean(controls),
            "af2_mean": statistics.mean(candidates),
            "delta_mean": statistics.mean(
                candidate - control for control, candidate in zip(controls, candidates)
            ),
        }

    bootstrap = _paired_parent_bootstrap(
        observations,
        candidate_arm="AF2",
        iterations=bootstrap_iterations,
        seed=20260821,
    )
    macro = aggregate["macro_map50_95"]
    direction_positive = bool(
        macro["delta_mean"] > 0 and macro["improved_seeds"] >= 2
    )
    payload = {
        "format": "coffee_detector.af2.reused_test_posthoc.v1",
        "protocol": "faruq-v3-af2-reused-test-posthoc-v1",
        "scientific_status": "REUSED_TEST_POSTHOC_NOT_LOCKED_CONFIRMATION",
        "seeds": list(SEEDS),
        "test_manifest_sha256": manifest_hash,
        "training_executed": False,
        "test_images_accessed": True,
        "further_tuning_authorized": False,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "classwise": classwise,
        "paired_parent_bootstrap": bootstrap,
        "status": (
            "POSTHOC_DIRECTION_POSITIVE"
            if direction_positive
            else "POSTHOC_DIRECTION_NOT_POSITIVE"
        ),
        "claim_limit": (
            "test was previously consumed by ACMC; descriptive post-hoc evidence only"
        ),
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    payload["summary"] = str(destination)
    return payload


def run_af2_reused_test(
    test_root: str | Path,
    d0ft_reports: list[str | Path],
    af2_checkpoints: list[str | Path],
    output_root: str | Path,
    *,
    device: str = "0",
    authorize_reused_test: bool = False,
) -> dict:
    if not authorize_reused_test:
        raise RuntimeError("Reused test memerlukan otorisasi eksplisit")
    if len(d0ft_reports) != 3 or len(af2_checkpoints) != 3:
        raise ValueError("D0FT report dan AF2 checkpoint masing-masing harus tiga")
    root = Path(test_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    manifest = root / "faruq_locked_test_manifest.json"
    data_yaml = root / "data.yaml"
    if not manifest.is_file() or not data_yaml.is_file():
        raise FileNotFoundError("Paket Faruq test tidak lengkap")
    manifest_hash = _sha256(manifest)
    d0ft_paths = [Path(path).expanduser().resolve() for path in d0ft_reports]
    checkpoints = [Path(path).expanduser().resolve() for path in af2_checkpoints]
    for path in d0ft_paths + checkpoints:
        if not path.is_file():
            raise FileNotFoundError(path)

    contract = {
        "format": "coffee_detector.af2.reused_test_contract.v1",
        "scientific_status": "REUSED_TEST_POSTHOC_NOT_LOCKED_CONFIRMATION",
        "test_manifest_sha256": manifest_hash,
        "d0ft_report_sha256": [_sha256(path) for path in d0ft_paths],
        "af2_checkpoint_sha256": [_sha256(path) for path in checkpoints],
        "seeds": list(SEEDS),
        "training_executed": False,
        "further_tuning_authorized": False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    contract_path = destination / "input_contract.json"
    if contract_path.is_file():
        if _load_json(contract_path, "Input contract") != contract:
            raise RuntimeError("Kontrak reused-test berubah; gunakan output baru")
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    runtime_yaml = _ultralytics_test_yaml(data_yaml, destination)
    reports_root = destination / "reports"
    af2_reports = []
    for index, seed in enumerate(SEEDS):
        report_path = reports_root / f"AF2_seed{seed}_reused_test.json"
        print(f"REUSED TEST AF2 seed={seed}", flush=True)
        report = _evaluate_checkpoint(
            checkpoints[index],
            runtime_yaml,
            root,
            report_path,
            checkpoint_hash=contract["af2_checkpoint_sha256"][index],
            test_manifest_hash=manifest_hash,
            device=device,
            protocol="faruq-v3-af2-reused-test-posthoc-report-v1",
        )
        _validate_test_report(
            report,
            manifest_hash=manifest_hash,
            require_protocol="faruq-v3-af2-reused-test-posthoc-report-v1",
        )
        af2_reports.append(report_path)

    return build_af2_reused_test_summary(
        d0ft_paths,
        af2_reports,
        manifest,
        destination / "af2_reused_test_posthoc_summary.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AF2 reused Faruq-test post-hoc evaluation")
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--d0ft-reports", nargs=3, required=True)
    parser.add_argument("--af2-checkpoints", nargs=3, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-reused-test", action="store_true")
    args = parser.parse_args()
    result = run_af2_reused_test(
        args.test_root,
        args.d0ft_reports,
        args.af2_checkpoints,
        args.output_root,
        device=args.device,
        authorize_reused_test=args.authorize_reused_test,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "aggregate": result["aggregate"],
                "paired_parent_bootstrap": result["paired_parent_bootstrap"],
                "scientific_status": result["scientific_status"],
            },
            indent=2,
        )
    )
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
