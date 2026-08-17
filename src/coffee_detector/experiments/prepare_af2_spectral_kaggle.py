from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from coffee_detector.af2_spectral.audit import sha256
from coffee_detector.experiments.prepare_faruq_v3_kaggle import (
    ARCHIVE_NAME,
    prepare_faruq_v3_kaggle_input,
)


PROJECT_ARTIFACTS = {
    ARCHIVE_NAME: "bundles/faruq-development-v3-grouped.tar",
    "D0_seed42_best.pt": "experiments/faruq-v3-yolo26n-baseline-v1/D0_seed42/weights/best.pt",
    "D0_seed123_best.pt": "experiments/faruq-v3-acmc-paired-confirmation-v1/D0_base/D0_seed123/weights/best.pt",
    "D0_seed2026_best.pt": "experiments/faruq-v3-acmc-paired-confirmation-v1/D0_base/D0_seed2026/weights/best.pt",
    "lfdet_afab_seed42_screening.json": "experiments/faruq-v3-breadth-screening-batch-v1/candidates/AFAB/val_reports/lfdet_afab_seed42_screening.json",
    "af2_igem_paired_confirmation.json": "experiments/faruq-v3-af2-igem-paired-confirmation-v1/val_reports/af2_igem_paired_confirmation.json",
}
MANIFEST_NAME = "af2_spectral_kaggle_manifest.json"


def build_af2_spectral_kaggle_bundle(
    project_root: str | Path,
    output_root: str | Path,
) -> dict:
    project_root = Path(project_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, relative in PROJECT_ARTIFACTS.items():
        source = project_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Artefak proyek tidak ditemukan: {source}")
        target = output_root / name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        artifacts[name] = {
            "source": relative,
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        }
    manifest = {
        "format": "coffee_detector.af2_spectral.kaggle_manifest.v1",
        "artifacts": artifacts,
        "test_images_included": False,
    }
    path = output_root / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(path)
    return manifest


def prepare_af2_spectral_kaggle_input(
    input_root: str | Path,
    work_root: str | Path,
) -> tuple[Path, dict[str, Path], dict]:
    input_root = Path(input_root).expanduser().resolve()
    work_root = Path(work_root).expanduser().resolve()
    matches = sorted(input_root.rglob(MANIFEST_NAME))
    if len(matches) != 1:
        raise FileNotFoundError(f"Harus ada satu {MANIFEST_NAME}; ditemukan {matches}")
    manifest = json.loads(matches[0].read_text(encoding="utf-8"))
    if manifest.get("format") != "coffee_detector.af2_spectral.kaggle_manifest.v1":
        raise RuntimeError("Format manifest Kaggle tidak dikenal")
    if manifest.get("test_images_included") is not False:
        raise RuntimeError("Bundle Kaggle tidak mempertahankan test lock")
    resolved = {}
    for name, contract in manifest.get("artifacts", {}).items():
        candidates = sorted(input_root.rglob(name))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Harus ada satu {name}; ditemukan {candidates}")
        path = candidates[0]
        if path.stat().st_size != int(contract["bytes"]) or sha256(path) != contract["sha256"]:
            raise RuntimeError(f"Kontrak SHA/ukuran gagal: {name}")
        resolved[name] = path
    if set(resolved) != set(PROJECT_ARTIFACTS):
        raise RuntimeError("Manifest tidak memuat kontrak artefak lengkap")
    data_root, dataset_contract = prepare_faruq_v3_kaggle_input(input_root, work_root)
    contract = {
        "format": "coffee_detector.af2_spectral.kaggle_input.v1",
        "dataset": dataset_contract,
        "artifacts": {name: str(path) for name, path in resolved.items()},
        "manifest_sha256": sha256(matches[0]),
        "test_images_accessed": False,
        "decision": "PASS",
    }
    (work_root / "af2_spectral_input_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    return data_root, resolved, contract


def restore_spectral_kaggle_run(
    input_root: str | Path,
    output_root: str | Path,
    *,
    arm: str,
    seed: int,
    d0_checkpoint: str | Path,
    config: str | Path,
) -> Path | None:
    """Restore one exact prior Kaggle output; never guess from a bare best.pt."""

    input_root = Path(input_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    expected = {
        "format": "coffee_detector.af2_spectral.run_contract.v1",
        "arm": arm,
        "seed": seed,
        "config_sha256": sha256(config),
        "d0_checkpoint_sha256": sha256(d0_checkpoint),
        "epochs": 50,
        "test_images_accessed": False,
    }
    matches = []
    for contract_path in input_root.rglob("run_contract.json"):
        try:
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload == expected:
            matches.append(contract_path.parent)
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Output resume {arm} seed {seed} ambigu: {matches}")
    destination = output_root / arm / f"{arm}_seed{seed}"
    if destination.exists():
        existing = destination / "run_contract.json"
        if not existing.is_file() or json.loads(existing.read_text(encoding="utf-8")) != expected:
            raise RuntimeError(f"Destination resume berisi kontrak lain: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(matches[0], destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/validate AF2 spectral Kaggle bundle")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--project-root", required=True)
    build.add_argument("--output-root", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--input-root", required=True)
    validate.add_argument("--work-root", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_af2_spectral_kaggle_bundle(args.project_root, args.output_root)
    else:
        _, _, result = prepare_af2_spectral_kaggle_input(args.input_root, args.work_root)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
