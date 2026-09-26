import ast
import json
from pathlib import Path

import pytest
import torch
import yaml

from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import METRICS
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_cwcf import (
    PROTOCOL as CWCF1_PROTOCOL,
)
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_cwcf_hyb import (
    PROTOCOL,
    _cwcf_without_representation,
    build_decision,
)
from coffee_detector.j25_cwcf import CWCFConfig, chromatic_wavelet_cue


ROOT = Path(__file__).resolve().parents[1]


def test_hyb1_changes_only_wavelet_representation_contract():
    candidate = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8N_CWCF_HYB1.yaml").read_text(
            encoding="utf-8"
        )
    )
    baseline = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8N_CWCF1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert candidate["dataset"] == baseline["dataset"]
    assert candidate["model"] == baseline["model"]
    assert candidate["weights"] == baseline["weights"]
    assert candidate["sampler"] == baseline["sampler"] == "none"
    assert candidate["train"] == baseline["train"]
    assert _cwcf_without_representation(candidate["cwcf"]) == _cwcf_without_representation(
        baseline["cwcf"]
    )

    base_frozen = CWCFConfig.from_mapping(baseline["cwcf"])
    hyb_frozen = CWCFConfig.from_mapping(candidate["cwcf"])
    assert base_frozen.wavelet_detail_mode == "energy"
    assert base_frozen.cue_channels == 4
    assert hyb_frozen.wavelet_detail_mode == "hybrid"
    assert hyb_frozen.cue_channels == 10


def test_hybrid_cue_exactly_contains_energy_and_directional_representations():
    image = torch.linspace(0.0, 1.0, 3 * 32 * 32).reshape(1, 3, 32, 32)
    energy = chromatic_wavelet_cue(image, detail_mode="energy")
    directional = chromatic_wavelet_cue(image, detail_mode="directional")
    hybrid = chromatic_wavelet_cue(image, detail_mode="hybrid")

    assert energy.shape == (1, 4, 32, 32)
    assert directional.shape == (1, 8, 32, 32)
    assert hybrid.shape == (1, 10, 32, 32)
    assert torch.equal(hybrid[:, :4], energy)
    assert torch.equal(hybrid[:, :2], directional[:, :2])
    assert torch.equal(hybrid[:, 4:], directional[:, 2:])


def test_invalid_cue_width_for_hybrid_mode_is_rejected():
    with pytest.raises(ValueError):
        CWCFConfig.from_mapping(
            {
                "cue_channels": 8,
                "wavelet_detail_mode": "hybrid",
            }
        )


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(macro, bottom, worst):
    return dict(zip(METRICS, (macro, bottom, worst)))


def test_hyb1_decision_compares_against_cwcf1(tmp_path):
    baseline = tmp_path / "cwcf1.json"
    candidate = tmp_path / "hyb1.json"
    common = {"seed": 42, "test_images_accessed": False}
    _write(
        baseline,
        {
            **common,
            "protocol": CWCF1_PROTOCOL,
            "metrics": _metrics(.7098, .3339, .0473),
            "target_class_map50_95": .0473,
        },
    )
    _write(
        candidate,
        {
            **common,
            "protocol": PROTOCOL,
            "metrics": _metrics(.7180, .3450, .0520),
            "target_class_map50_95": .0520,
        },
    )
    result = build_decision(baseline, candidate, tmp_path / "decision.json")
    assert result["decision"] == "PASS_HYB1_SCREEN"
    assert result["v8n_cwcf_hyb1_minus_cwcf1"]["macro_map50_95"] == pytest.approx(
        .0082
    )
    assert result["v8n_cwcf_hyb1_minus_cwcf1"][
        "bottom3_class_map50_95"
    ] == pytest.approx(.0111)
    assert result["target_delta"] == pytest.approx(.0047)
    assert result["test_opened"] is False


def test_protocol_and_notebook_keep_single_change_and_test_lock():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_YOLOV8N_CWCF_HYB1_PROTOCOL_2026-09-26.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before training**" in protocol
    assert "[Cb, Cr, D1, D2, LH1, HL1, HH1, LH2, HL2, HH2]" in protocol
    assert "attribute_gain = 0.15" in protocol
    assert "PASS_HYB1_SCREEN" in protocol

    notebook = json.loads(
        (
            ROOT / "notebooks/Coffee_Standard_J25_YOLOv8n_CWCF_HYB1_Seed42_Colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-yolov8n-cwcf-hyb1'" in code
    assert "V8N_CWCF1_seed42_result.json" in code
    assert "run_coffee_standard_j25_yolov8n_cwcf_hyb" in code
    assert "--authorize-training" in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
