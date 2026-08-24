import json
from pathlib import Path

from coffee_detector.experiments.run_faruq_v3_af2_iso_arm import ALLOWED_SEEDS
from coffee_detector.experiments.run_faruq_v3_af2_orient_confirmation_decision import (
    METRICS,
    run_faruq_v3_af2_orient_confirmation_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metrics(values):
    return dict(zip(METRICS, values))


def test_only_orientation_is_authorized_for_confirmation_seeds():
    assert ALLOWED_SEEDS["AF2_RADIAL"] == (42,)
    assert ALLOWED_SEEDS["AF2_ORIENT"] == (42, 123, 2026)


def test_orientation_paired_gate_passes_tail_gain(tmp_path):
    af2 = {
        "42": _metrics((0.8820, 0.8004, 0.7935)),
        "123": _metrics((0.8822, 0.7822, 0.7646)),
        "2026": _metrics((0.8740, 0.7985, 0.7865)),
    }
    baseline = _write(
        tmp_path / "af2.json",
        {
            "protocol": "faruq-v3-af2-igem-paired-validation-confirmation-v1",
            "seeds": [42, 123, 2026],
            "test_images_accessed": False,
            "test_opened": False,
            "decisions": {"AF2": {"decision": "PASS"}},
            "per_seed": {seed: {"AF2": value} for seed, value in af2.items()},
        },
    )
    results = []
    values = {
        42: (0.8833, 0.8138, 0.8014),
        123: (0.8840, 0.7910, 0.7720),
        2026: (0.8750, 0.8070, 0.7900),
    }
    for seed, metrics in values.items():
        results.append(
            _write(
                tmp_path / f"orient-{seed}.json",
                {
                    "format": "coffee_detector.af2_iso.arm_result.v1",
                    "arm": "AF2_ORIENT",
                    "seed": seed,
                    "evaluation_split": "val",
                    "test_images_accessed": False,
                    "metrics": _metrics(metrics),
                    "checkpoint_sha256": f"orient-{seed}",
                    "initial_d0_checkpoint_sha256": f"d0-{seed}",
                },
            )
        )
    result = run_faruq_v3_af2_orient_confirmation_decision(
        baseline, *results, tmp_path / "decision.json"
    )
    assert result["decision"] == "PASS"
    assert result["next"] == "RETAIN_AF2_ORIENT"
    assert result["test_opened"] is False


def test_protocol_freezes_tail_focused_gate():
    protocol = (
        ROOT / "docs/FARUQ_V3_AF2_ORIENT_PAIRED_CONFIRMATION_PROTOCOL_2026-08-21.md"
    ).read_text(encoding="utf-8")
    assert "frozen before seed 123/2026 training" in protocol
    assert "mean Bottom-3 gain is at least +0.5" in protocol
    assert "Test: locked" in protocol

    notebook = json.loads(
        (ROOT / "notebooks/Faruq_V3_AF2_ORIENT_Confirmation_Arm_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "SEED = 123" in source
    assert "D0_base/D0_seed{SEED}/weights/best.pt" in source
    assert "AF2_ORIENT seed {SEED}: {epochs}/50" in source
    assert "test').exists()" in source
