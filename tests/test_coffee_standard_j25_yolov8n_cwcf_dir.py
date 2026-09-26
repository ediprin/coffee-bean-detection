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
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_cwcf_dir import (
    PROTOCOL,
    _cwcf_without_representation,
    build_decision,
)
from coffee_detector.j25_cwcf import (
    CWCFConfig,
    chromatic_wavelet_cue,
    haar_bands,
    haar_decompose,
)


ROOT = Path(__file__).resolve().parents[1]


def test_dir1_changes_only_wavelet_representation_contract():
    candidate = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8N_CWCF_DIR1.yaml").read_text(
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
    dir_frozen = CWCFConfig.from_mapping(candidate["cwcf"])
    assert base_frozen.wavelet_detail_mode == "energy"
    assert base_frozen.cue_channels == 4
    assert dir_frozen.wavelet_detail_mode == "directional"
    assert dir_frozen.cue_channels == 8


def test_directional_cue_preserves_chroma_and_exposes_six_detail_bands():
    image = torch.linspace(0.0, 1.0, 3 * 32 * 32).reshape(1, 3, 32, 32)
    energy = chromatic_wavelet_cue(image, detail_mode="energy")
    directional = chromatic_wavelet_cue(image, detail_mode="directional")
    assert energy.shape == (1, 4, 32, 32)
    assert directional.shape == (1, 8, 32, 32)
    assert torch.equal(energy[:, :2], directional[:, :2])


def test_haar_energy_is_exactly_reconstructible_from_directional_bands():
    value = torch.linspace(-1.0, 1.0, 34 * 30).reshape(1, 1, 34, 30)
    ll, lh, hl, hh = haar_bands(value)
    ll_ref, detail = haar_decompose(value)
    reconstructed = torch.sqrt(
        torch.cat((lh, hl, hh), dim=1).square().mean(dim=1, keepdim=True) + 1e-8
    )
    assert torch.equal(ll, ll_ref)
    assert torch.equal(detail, reconstructed)


def test_invalid_cue_width_for_directional_mode_is_rejected():
    with pytest.raises(ValueError):
        CWCFConfig.from_mapping(
            {
                "cue_channels": 4,
                "wavelet_detail_mode": "directional",
            }
        )


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(macro, bottom, worst):
    return dict(zip(METRICS, (macro, bottom, worst)))


def test_dir1_decision_compares_against_cwcf1(tmp_path):
    baseline = tmp_path / "cwcf1.json"
    candidate = tmp_path / "dir1.json"
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
            "metrics": _metrics(.7180, .3450, .0460),
            "target_class_map50_95": .0460,
        },
    )
    result = build_decision(baseline, candidate, tmp_path / "decision.json")
    assert result["decision"] == "PASS_DIR1_SCREEN"
    assert result["v8n_cwcf_dir1_minus_cwcf1"]["macro_map50_95"] == pytest.approx(
        .0082
    )
    assert result["v8n_cwcf_dir1_minus_cwcf1"][
        "bottom3_class_map50_95"
    ] == pytest.approx(.0111)
    assert result["criteria"]["worst_class_is_descriptive_not_a_promotion_gate"] is True
    assert result["target_delta"] == pytest.approx(-.0013)
    assert result["test_opened"] is False


def test_protocol_and_notebook_keep_single_change_and_test_lock():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_YOLOV8N_CWCF_DIR1_PROTOCOL_2026-09-26.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before training**" in protocol
    assert "[Cb, Cr, LH1, HL1, HH1, LH2, HL2, HH2]" in protocol
    assert "attribute_gain = 0.15" in protocol
    assert "PASS_DIR1_SCREEN" in protocol

    notebook = json.loads(
        (
            ROOT / "notebooks/Coffee_Standard_J25_YOLOv8n_CWCF_DIR1_Seed42_Colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-yolov8n-cwcf-dir1'" in code
    assert "V8N_CWCF1_seed42_result.json" in code
    assert "run_coffee_standard_j25_yolov8n_cwcf_dir" in code
    assert "--authorize-training" in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
