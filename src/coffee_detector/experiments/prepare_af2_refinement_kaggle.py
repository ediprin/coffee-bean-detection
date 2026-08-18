from __future__ import annotations

import json
import shutil
from pathlib import Path

from coffee_detector.af2_spectral.audit import sha256
from coffee_detector.experiments.prepare_af2_spectral_kaggle import (
    prepare_af2_spectral_kaggle_input,
)


def prepare_af2_refinement_kaggle_input(
    input_root: str | Path, work_root: str | Path
) -> tuple[Path, dict[str, Path], dict]:
    """Reuse the already validated private AF2 core bundle without opening test."""

    data_root, artifacts, parent = prepare_af2_spectral_kaggle_input(
        input_root, work_root
    )
    contract = {
        "format": "coffee_detector.af2_refinement.kaggle_input.v1",
        "parent_contract": parent,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "test_images_accessed": False,
        "decision": "PASS",
    }
    target = Path(work_root).expanduser().resolve() / "af2_refinement_input_contract.json"
    target.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return data_root, artifacts, contract


def restore_af2_refinement_run(
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
        "format": "coffee_detector.af2_refinement.run_contract.v1",
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
        if not existing.is_file() or json.loads(
            existing.read_text(encoding="utf-8")
        ) != expected:
            raise RuntimeError(f"Destination resume berisi kontrak lain: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(matches[0], destination)
    return destination
