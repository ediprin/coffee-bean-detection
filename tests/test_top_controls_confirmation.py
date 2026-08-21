import json
import hashlib
import shutil
from pathlib import Path

from coffee_detector.experiments.run_faruq_v3_top_controls_confirmation_decision import (
    METRICS,
    run_faruq_v3_top_controls_confirmation_decision,
)
from coffee_detector.experiments.prepare_top_controls_kaggle import (
    ARTIFACTS,
    CORE_MANIFEST_FORMAT,
    build_top_controls_kaggle_bundle,
    build_top_controls_canonical_kaggle_core,
    ensure_run_contract,
    restore_top_control_kaggle_run,
    run_directory,
)
from coffee_detector.experiments import prepare_top_controls_kaggle as kaggle_core


ROOT = Path(__file__).resolve().parents[1]


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metrics(macro, bottom3, worst):
    return dict(zip(METRICS, (macro, bottom3, worst)))


def _arm(path, arm, seed, values, format_name):
    return _write(
        path,
        {
            "format": format_name,
            "arm": arm,
            "seed": seed,
            "metrics": values,
            "test_images_accessed": False,
        },
    )


def test_top_control_decision_uses_paired_seed_matched_controls(tmp_path):
    seeds = [42, 123, 2026]
    stb_values = {str(seed): _metrics(0.88, 0.80, 0.78) for seed in seeds}
    af2_values = {str(seed): _metrics(0.88, 0.79, 0.77) for seed in seeds}
    continuation_values = {str(seed): _metrics(0.87, 0.78, 0.76) for seed in seeds}
    stb = _write(
        tmp_path / "stb.json",
        {
            "protocol": "faruq-v3-stb-capacity-paired-confirmation-v1",
            "seeds": seeds,
            "test_images_accessed": False,
            "test_opened": False,
            "per_seed": {seed: {"STB1": values} for seed, values in stb_values.items()},
        },
    )
    af2 = _write(
        tmp_path / "af2.json",
        {
            "protocol": "faruq-v3-af2-igem-paired-validation-confirmation-v1",
            "seeds": seeds,
            "test_images_accessed": False,
            "test_opened": False,
            "per_seed": {seed: {"AF2": values} for seed, values in af2_values.items()},
        },
    )
    continuation = _write(
        tmp_path / "continuation.json",
        {
            "format": "coffee_detector.af2_continuation.paired_confirmation.v1",
            "seeds": seeds,
            "test_images_accessed": False,
            "test_opened": False,
            "per_seed": {
                seed: {"AF2CT30": values}
                for seed, values in continuation_values.items()
            },
        },
    )
    fct42 = _write(tmp_path / "fct42.json", {"metrics": _metrics(0.89, 0.84, 0.83)})
    af2r42 = _write(
        tmp_path / "af2r42.json",
        {
            "format": "coffee_detector.af2r.seed42_recovered_evidence.v1",
            "test_opened": False,
            "values": {
                "AF2R0": _metrics(0.90, 0.84, 0.83),
                "AF2R1": _metrics(0.89, 0.83, 0.82),
            },
        },
    )
    af2cal42 = _write(
        tmp_path / "af2cal42.json",
        {
            "format": "coffee_detector.af2cal.frozen_evidence.v1",
            "test_opened": False,
            "values": {"AF2CAL3": _metrics(0.869, 0.779, 0.759)},
        },
    )

    paths = {arm: [] for arm in ("FCT0", "AF2R0", "AF2R1", "AF2CAL3")}
    formats = {
        "FCT0": "coffee_detector.fct0_confirmation.arm_result.v1",
        "AF2R0": "coffee_detector.af2r.arm_result.v1",
        "AF2R1": "coffee_detector.af2r.arm_result.v1",
        "AF2CAL3": "coffee_detector.af2cal.arm_result.v1",
    }
    values = {
        "FCT0": _metrics(0.89, 0.82, 0.80),
        "AF2R0": _metrics(0.90, 0.82, 0.81),
        "AF2R1": _metrics(0.89, 0.81, 0.80),
        "AF2CAL3": _metrics(0.869, 0.779, 0.759),
    }
    for arm in paths:
        for seed in (123, 2026):
            paths[arm].append(
                _arm(tmp_path / f"{arm}_{seed}.json", arm, seed, values[arm], formats[arm])
            )

    result = run_faruq_v3_top_controls_confirmation_decision(
        stb,
        af2,
        continuation,
        fct42,
        af2r42,
        af2cal42,
        tuple(paths["FCT0"]),
        tuple(paths["AF2R0"]),
        tuple(paths["AF2R1"]),
        tuple(paths["AF2CAL3"]),
        tmp_path / "decision.json",
    )
    assert result["comparisons"]["FCT0"]["decision"] == "PASS"
    assert result["comparisons"]["AF2R0"]["decision"] == "PASS"
    assert result["comparisons"]["AF2R1"]["decision"] == "FAIL"
    assert result["comparisons"]["AF2CAL3"]["decision"] == "FAIL"
    assert result["test_opened"] is False
    assert result["note"] == "AF2FT30 is reused as AF2CT30; it is not retrained."


def test_protocol_freezes_all_unconfirmed_high_seed42_arms():
    protocol = (
        ROOT / "docs/FARUQ_V3_TOP_CONTROLS_PAIRED_CONFIRMATION_PROTOCOL_2026-08-21.md"
    ).read_text(encoding="utf-8")
    for arm in ("FCT0", "AF2R0", "AF2R1", "AF2CAL3"):
        assert f"`{arm}`" in protocol
    assert "`AF2FT30` is not retrained" in protocol
    assert "Test: locked" in protocol


def test_af2_family_runners_require_seed_matched_checkpoints():
    for relative in (
        "src/coffee_detector/experiments/run_faruq_v3_af2r_arm.py",
        "src/coffee_detector/experiments/run_faruq_v3_af2cal_arm.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "ALLOWED_SEEDS = (42, 123, 2026)" in source
        assert "Checkpoint AF2 bukan seed {seed}" in source


def test_colab_notebook_is_parameterized_resumable_and_test_locked():
    notebook = json.loads(
        (ROOT / "notebooks/Faruq_V3_Top_Controls_Multiseed_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for arm in ("FCT0", "AF2R0", "AF2R1", "AF2CAL3"):
        assert arm in source
    assert "SEED = 123" in source
    assert "assert SEED in {123,2026}" in source
    assert "weights/best.pt" in source
    assert "RESUME" not in source or "START/RESUME" in source
    assert "last.pt" in source
    assert "assert not (DATA/'test').exists()" in source
    assert "test tidak boleh tersedia" in source
    assert "generic best.pt" not in source
    assert "top_controls_paired_confirmation.json" in source
    assert "sys.path.insert(0,str(SRC))" in source
    assert "importlib.invalidate_caches()" in source


def test_parallel_colab_notebooks_fix_one_unique_arm_and_seed():
    expected = {
        (arm, seed)
        for arm in ("FCT0", "AF2R0", "AF2R1", "AF2CAL3")
        for seed in (123, 2026)
    }
    observed = set()
    for arm, seed in sorted(expected):
        path = ROOT / f"notebooks/Faruq_V3_{arm}_Seed{seed}_Colab.ipynb"
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in payload["cells"]
        )
        for cell in payload["cells"]:
            if cell.get("cell_type") == "code":
                compile("".join(cell["source"]), str(path), "exec")
        assert f"ARM = '{arm}'   # fixed parallel arm" in source
        assert f"SEED = {seed}     # fixed parallel seed" in source
        assert "sys.path.insert(0,str(SRC))" in source
        assert "assert not (DATA/'test').exists()" in source
        observed.add((arm, seed))
    assert observed == expected


def test_kaggle_bundle_and_resume_contract_are_explicit(tmp_path):
    project = tmp_path / "project"
    for relative in ARTIFACTS.values():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact:{relative}".encode())
    bundle = tmp_path / "bundle"
    manifest = build_top_controls_kaggle_bundle(project, bundle)
    assert manifest["test_images_included"] is False
    assert set(manifest["artifacts"]) == set(ARTIFACTS)

    output = tmp_path / "output"
    source = bundle / "AF2_seed123_best.pt"
    contract = ensure_run_contract(
        output, arm="AF2R0", seed=123, source_checkpoint=source
    )
    (contract.parent / "weights").mkdir()
    (contract.parent / "weights" / "last.pt").write_bytes(b"resume")
    saved = tmp_path / "saved-version"
    shutil.copytree(output, saved)
    restored_output = tmp_path / "restored"
    restored = restore_top_control_kaggle_run(
        saved,
        restored_output,
        arm="AF2R0",
        seed=123,
        source_checkpoint=source,
    )
    assert restored == run_directory(restored_output, "AF2R0", 123)
    assert (restored / "weights/last.pt").read_bytes() == b"resume"


def test_all_sequential_kaggle_notebook_compiles_and_stays_test_locked():
    path = ROOT / "notebooks/Faruq_V3_Top_Controls_All_Sequential_Kaggle.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), str(path), "exec")
    assert "ARMS=('FCT0','AF2R0','AF2R1','AF2CAL3')" in source
    assert "SEEDS=(123,2026)" in source
    assert "restore_top_control_kaggle_run" in source
    assert "faruq-development-v3-grouped.tar.bin" in source
    assert "top_controls_kaggle_manifest.json" in source
    assert "assert not (DATA/'test').exists()" in source
    assert "test_images_accessed':False" in source
    assert "make_archive" in source


def test_canonical_kaggle_core_is_opaque_load_tested_and_excludes_runs(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    for relative in ARTIFACTS.values():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact:{relative}".encode())
    archive = project / "bundles/faruq-development-v3-grouped.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"opaque canonical archive")
    monkeypatch.setattr(kaggle_core, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(
        kaggle_core, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        kaggle_core,
        "_validate_yolo_checkpoint",
        lambda path, seed: {
            "loadable_by_ultralytics": True,
            "seed": seed,
            "nc": 21,
            "parameters": 1,
            "bytes": path.stat().st_size,
            "sha256": kaggle_core.sha256(path),
        },
    )

    bundle = tmp_path / "bundle"
    manifest = build_top_controls_canonical_kaggle_core(project, bundle)
    assert manifest["format"] == CORE_MANIFEST_FORMAT
    assert manifest["archive"]["opaque"] is True
    assert manifest["completed_training_runs_included"] is False
    assert manifest["resume_source"] == "kaggle_saved_version_only"
    assert manifest["test_images_included"] is False
    assert not list(bundle.rglob("run_contract.json"))


def test_canonical_upload_colab_follows_historical_core_protocol():
    path = (
        ROOT
        / "notebooks/Faruq_V3_Top_Controls_All_In_One_Kaggle_Upload_Colab.ipynb"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), str(path), "exec")
    assert "build_top_controls_canonical_kaggle_core" in source
    assert "faruq-v3-experiment-core-v1" in source
    assert "faruq-development-v3-grouped.tar.bin" in source
    assert "ARCHIVE_SHA256" in source
    assert "EXPECTED_ANNOTATIONS" in source
    assert "checkpoint_validation" in source
    assert "completed_training_runs_included" in source
    assert "kaggle_saved_version_only" in source
    assert "run_contract.json" in source
    assert "test_images_included" in source
