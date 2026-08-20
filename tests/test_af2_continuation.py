import json
from pathlib import Path

import yaml

from coffee_detector.experiments.run_faruq_v3_af2_continuation_arm import (
    _baseline_metrics,
)
from coffee_detector.experiments.run_faruq_v3_af2_continuation_decision import (
    METRICS,
    run_faruq_v3_af2_continuation_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metrics(values):
    return dict(zip(METRICS, values))


def test_continuation_config_preserves_frozen_af2_schedule():
    continuation = yaml.safe_load(
        (ROOT / "configs/af2_continuation/AF2CT30_yolo26n.yaml").read_text()
    )
    historical = yaml.safe_load(
        (ROOT / "configs/af2cal/AF2FT30_yolo26n.yaml").read_text()
    )
    assert continuation["afab"] == historical["afab"]
    assert continuation["train"] == historical["train"]
    assert continuation["mechanism"] == "af2_continuation_only"


def test_baseline_confirmation_is_validation_only():
    payload = {
        "protocol": "faruq-v3-af2-igem-paired-validation-confirmation-v1",
        "seeds": [42, 123, 2026],
        "test_images_accessed": False,
        "test_opened": False,
        "decisions": {"AF2": {"decision": "PASS"}},
        "per_seed": {"123": {"AF2": _metrics((0.88, 0.78, 0.76))}},
    }
    assert _baseline_metrics(payload, 123)["macro_map50_95"] == 0.88


def test_paired_decision_passes_stable_continuation(tmp_path):
    baselines = {
        "42": _metrics((0.882, 0.800, 0.793)),
        "123": _metrics((0.882, 0.782, 0.765)),
        "2026": _metrics((0.874, 0.798, 0.787)),
    }
    confirmation = _write(
        tmp_path / "confirmation.json",
        {
            "protocol": "faruq-v3-af2-igem-paired-validation-confirmation-v1",
            "seeds": [42, 123, 2026],
            "test_images_accessed": False,
            "test_opened": False,
            "decisions": {"AF2": {"decision": "PASS"}},
            "per_seed": {seed: {"AF2": value} for seed, value in baselines.items()},
        },
    )
    evidence = _write(
        tmp_path / "evidence.json",
        {
            "format": "coffee_detector.af2cal.frozen_evidence.v1",
            "seed": 42,
            "test_opened": False,
            "values": {"AF2": baselines["42"], "AF2FT30": _metrics((0.890, 0.839, 0.835))},
        },
    )
    paths = []
    for seed, candidate in ((123, (0.891, 0.800, 0.790)), (2026, (0.881, 0.810, 0.800))):
        paths.append(
            _write(
                tmp_path / f"seed{seed}.json",
                {
                    "format": "coffee_detector.af2_continuation.arm_result.v1",
                    "seed": seed,
                    "test_images_accessed": False,
                    "baseline_af2_metrics": baselines[str(seed)],
                    "metrics": _metrics(candidate),
                    "initial_af2_checkpoint_sha256": f"initial-{seed}",
                    "checkpoint_sha256": f"final-{seed}",
                },
            )
        )
    result = run_faruq_v3_af2_continuation_decision(
        confirmation, evidence, paths[0], paths[1], tmp_path / "decision.json"
    )
    assert result["decision"] == "PASS"
    assert result["next"] == "RETAIN_AF2_TWO_STAGE_PROTOCOL"
    assert result["test_opened"] is False


def test_protocol_is_frozen_and_does_not_open_test():
    text = (
        ROOT / "docs/FARUQ_V3_AF2_CONTINUATION_PAIRED_CONFIRMATION_PROTOCOL_2026-08-21.md"
    ).read_text(encoding="utf-8")
    assert "frozen before seed 123/2026 training" in text
    assert "Test: locked" in text
    assert "do not retrain" in text

    notebook = json.loads(
        (ROOT / "notebooks/Faruq_V3_AF2_Continuation_Arm_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "SEED = 123" in source
    assert "shutil.copy" not in source  # resume is handled by the runner, never copied ad hoc
    assert "test tidak boleh tersedia" in source
    assert "AF2CT30 seed {SEED}: {epochs}/30" in source
