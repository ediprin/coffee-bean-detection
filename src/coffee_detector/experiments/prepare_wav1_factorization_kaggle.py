from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from coffee_detector.experiments.prepare_af2_spectral_kaggle import (
    prepare_af2_spectral_kaggle_input,
)
from coffee_detector.wav1_factorization.audit import sha256


ADDON_MANIFEST_NAME = "wav1_factorization_kaggle_manifest.json"
ADDON_MANIFEST_FORMAT = "coffee_detector.wav1_factorization.kaggle_addon_manifest.v1"
EXPECTED_WAV1_SEED42 = {
    "macro_map50_95": 0.8841052369918866,
    "bottom3_class_map50_95": 0.8327607439278027,
    "worst_class_map50_95": 0.8203489485589485,
}


def _read_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _validate_wav1_seed42(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    payload = _read_json(source, "WAV1 seed42 result")
    if (
        payload.get("format") != "coffee_detector.af2_spectral.arm_result.v1"
        or payload.get("arm") != "WAV1"
        or int(payload.get("seed", -1)) != 42
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
    ):
        raise RuntimeError("WAV1 seed42 bukan result validation-only yang dibekukan")
    metrics = payload.get("metrics", {})
    for key, expected in EXPECTED_WAV1_SEED42.items():
        if abs(float(metrics.get(key, float("nan"))) - expected) > 1.0e-12:
            raise RuntimeError(
                f"WAV1 seed42 berubah dari reference frozen: {key}={metrics.get(key)} != {expected}"
            )
    return payload


def _find_wav1_seed42(project_root: Path) -> Path:
    candidates: list[Path] = []
    for path in project_root.rglob("WAV1_seed42_result.json"):
        try:
            _validate_wav1_seed42(path)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
        candidates.append(path.resolve())
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Harus ditemukan tepat satu WAV1_seed42_result.json yang cocok dengan frozen reference; "
            f"ditemukan {candidates}"
        )
    return candidates[0]


def build_wav1_factorization_kaggle_addon(
    project_root: str | Path,
    output_root: str | Path,
    *,
    wav1_seed42_result: str | Path | None = None,
) -> dict:
    """Build the small add-on dataset used beside the existing AF2 spectral core.

    The large Faruq development archive and D0 checkpoint are intentionally not
    duplicated. Kaggle must attach the existing AF2 spectral core plus this add-on.
    """

    project_root = Path(project_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source = (
        Path(wav1_seed42_result).expanduser().resolve()
        if wav1_seed42_result is not None
        else _find_wav1_seed42(project_root)
    )
    _validate_wav1_seed42(source)
    target = output_root / "WAV1_seed42_result.json"
    if source != target:
        shutil.copy2(source, target)
    if sha256(source) != sha256(target) or source.stat().st_size != target.stat().st_size:
        raise RuntimeError("Copy WAV1 seed42 result mengubah bytes/SHA")

    manifest = {
        "format": ADDON_MANIFEST_FORMAT,
        "artifacts": {
            "WAV1_seed42_result.json": {
                "source": str(source.relative_to(project_root))
                if project_root in source.parents
                else str(source),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        },
        "requires_existing_core_manifest": "af2_spectral_kaggle_manifest.json",
        "test_images_included": False,
    }
    manifest_path = output_root / ADDON_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def _d0ft_seed42_reference(artifacts: dict[str, Path], work_root: Path) -> Path:
    breadth = _read_json(
        artifacts["lfdet_afab_seed42_screening.json"],
        "AFAB breadth-screening reference",
    )
    if (
        breadth.get("protocol") != "faruq-v3-lfdet-afab-breadth-screening-v1"
        or int(breadth.get("seed", -1)) != 42
        or breadth.get("evaluation_split") != "val"
        or breadth.get("test_images_accessed") is not False
        or breadth.get("test_opened") is not False
    ):
        raise RuntimeError("AFAB breadth result bukan seed42 validation-only reference")
    d0ft = breadth.get("controls", {}).get("D0FT")
    if not isinstance(d0ft, dict):
        raise RuntimeError("AFAB breadth result tidak memuat controls.D0FT")
    # Keep the full metrics object when available so the later report can compare
    # per-class AP deltas rather than headline metrics only.
    payload = {
        "format": "coffee_detector.wav1_factorization.reference.v1",
        "arm": "D0FT",
        "seed": 42,
        "metrics": d0ft.get("metrics", d0ft),
        "evaluation_split": "val",
        "test_images_accessed": False,
        "source": str(artifacts["lfdet_afab_seed42_screening.json"]),
    }
    destination = work_root / "d0ft_seed42_reference.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def prepare_wav1_factorization_kaggle_input(
    input_root: str | Path,
    work_root: str | Path,
) -> tuple[Path, dict[str, Path], dict]:
    """Validate the existing core and the small WAV1 factorization add-on."""

    input_root = Path(input_root).expanduser().resolve()
    work_root = Path(work_root).expanduser().resolve()
    data_root, core_artifacts, core_contract = prepare_af2_spectral_kaggle_input(
        input_root, work_root
    )
    manifests = sorted(input_root.rglob(ADDON_MANIFEST_NAME))
    if len(manifests) != 1:
        raise FileNotFoundError(
            f"Harus ada satu {ADDON_MANIFEST_NAME}; ditemukan {manifests}"
        )
    addon = _read_json(manifests[0], "WAV1 factorization add-on manifest")
    if addon.get("format") != ADDON_MANIFEST_FORMAT:
        raise RuntimeError("Format WAV1 factorization add-on tidak cocok")
    if addon.get("test_images_included") is not False:
        raise RuntimeError("WAV1 factorization add-on melanggar test lock")

    artifact_contract = addon.get("artifacts", {}).get("WAV1_seed42_result.json")
    if not isinstance(artifact_contract, dict):
        raise RuntimeError("Add-on tidak memuat WAV1_seed42_result.json")
    candidates = sorted(input_root.rglob("WAV1_seed42_result.json"))
    candidates = [path for path in candidates if path != manifests[0]]
    valid = []
    for path in candidates:
        if (
            path.stat().st_size == int(artifact_contract["bytes"])
            and sha256(path) == artifact_contract["sha256"]
        ):
            try:
                _validate_wav1_seed42(path)
            except RuntimeError:
                continue
            valid.append(path.resolve())
    if len(valid) != 1:
        raise FileNotFoundError(f"WAV1 seed42 artifact add-on ambigu/tidak ada: {valid}")

    wav1_result = valid[0]
    d0ft_reference = _d0ft_seed42_reference(core_artifacts, work_root)
    resolved = dict(core_artifacts)
    resolved["WAV1_seed42_result.json"] = wav1_result
    resolved["D0FT_seed42_reference.json"] = d0ft_reference
    contract = {
        "format": "coffee_detector.wav1_factorization.kaggle_input.v1",
        "dataset": core_contract["dataset"],
        "core_manifest_sha256": core_contract["manifest_sha256"],
        "addon_manifest_sha256": sha256(manifests[0]),
        "artifacts": {name: str(path) for name, path in resolved.items()},
        "test_images_accessed": False,
        "decision": "PASS",
    }
    contract_path = work_root / "wav1_factorization_input_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return data_root, resolved, contract


def restore_wav1_factorization_kaggle_run(
    input_root: str | Path,
    output_root: str | Path,
    *,
    arm: str,
    seed: int,
    d0_checkpoint: str | Path,
    config: str | Path,
) -> Path | None:
    input_root = Path(input_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    expected = {
        "format": "coffee_detector.wav1_factorization.run_contract.v1",
        "arm": arm,
        "seed": seed,
        "config_sha256": sha256(config),
        "d0_checkpoint_sha256": sha256(d0_checkpoint),
        "epochs": 50,
        "test_images_accessed": False,
    }
    matches: list[Path] = []
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
        current = destination / "run_contract.json"
        if not current.is_file() or json.loads(current.read_text(encoding="utf-8")) != expected:
            raise RuntimeError(f"Destination resume memiliki kontrak lain: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(matches[0], destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/validate WAV1 factorization Kaggle add-on")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--project-root", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--wav1-seed42-result")
    validate = sub.add_parser("validate")
    validate.add_argument("--input-root", required=True)
    validate.add_argument("--work-root", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_wav1_factorization_kaggle_addon(
            args.project_root,
            args.output_root,
            wav1_seed42_result=args.wav1_seed42_result,
        )
    else:
        _, _, result = prepare_wav1_factorization_kaggle_input(
            args.input_root, args.work_root
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
