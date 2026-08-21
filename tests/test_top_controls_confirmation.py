import json
from pathlib import Path

from coffee_detector.experiments.run_faruq_v3_top_controls_confirmation_decision import (
    METRICS,
    run_faruq_v3_top_controls_confirmation_decision,
)


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
