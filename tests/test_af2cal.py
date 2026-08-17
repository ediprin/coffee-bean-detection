import json
from pathlib import Path

import torch
import yaml

from coffee_detector.af2cal import (
    AF2CalibratedDetectionModel,
    AF2ChannelCalibratedEnhancer,
)
from coffee_detector.afab import AFABConfig, AFABInputEnhancer
from coffee_detector.experiments.run_faruq_v3_af2cal_decision import (
    run_faruq_v3_af2cal_decision,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def afab_config():
    return AFABConfig(mode="af2", patch_size=32, overlap=0.5, chunk_size=8)


def test_zero_initialized_calibration_is_exact_af2():
    raw = torch.rand(1, 3, 64, 64)
    fixed = AFABInputEnhancer(afab_config())
    calibrated = AF2ChannelCalibratedEnhancer(afab_config())

    output, scale = calibrated.forward_with_scale(raw)

    assert torch.equal(scale, torch.ones_like(scale))
    assert torch.equal(output, fixed(raw))


def test_calibration_has_three_parameters_and_finite_gradients():
    raw = torch.rand(1, 3, 64, 64)
    calibrated = AF2ChannelCalibratedEnhancer(afab_config())
    assert sum(parameter.numel() for parameter in calibrated.parameters()) == 3
    calibrated.calibration_logits.data.fill_(0.25)

    output, scale = calibrated.forward_with_scale(raw)
    assert scale.min() >= 0 and scale.max() <= 2
    assert not torch.equal(output, calibrated.af2(raw))
    output.mean().backward()
    assert calibrated.calibration_logits.grad is not None
    assert torch.isfinite(calibrated.calibration_logits.grad).all()


def test_calibrated_detector_adds_only_three_parameters():
    from coffee_detector.afab import AFABDetectionModel

    fixed = AFABDetectionModel(
        str(MODEL_YAML), nc=5, verbose=False, afab=afab_config()
    )
    calibrated = AF2CalibratedDetectionModel(
        str(MODEL_YAML), nc=5, verbose=False, afab=afab_config()
    )
    fixed_parameters = sum(parameter.numel() for parameter in fixed.parameters())
    calibrated_parameters = sum(
        parameter.numel() for parameter in calibrated.parameters()
    )
    assert calibrated_parameters - fixed_parameters == 3
    extra = set(calibrated.state_dict()) - set(fixed.state_dict())
    assert extra == {"af2cal.calibration_logits"}


def test_af2cal_configs_are_schedule_matched():
    payloads = {
        path.stem.split("_")[0]: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in (ROOT / "configs/af2cal").glob("AF2*.yaml")
    }
    assert set(payloads) == {"AF2FT30", "AF2CAL3"}
    assert payloads["AF2FT30"]["model"] == payloads["AF2CAL3"]["model"]
    assert payloads["AF2FT30"]["afab"] == payloads["AF2CAL3"]["afab"]
    assert payloads["AF2FT30"]["train"] == payloads["AF2CAL3"]["train"]
    assert payloads["AF2FT30"]["mechanism"] == "continuation_only"
    assert payloads["AF2CAL3"]["mechanism"] == "channel_residual_calibration"


def test_af2cal_decision_requires_gain_over_ft_and_retention_of_r0(tmp_path):
    output = tmp_path / "output"
    reports = output / "val_reports"
    reports.mkdir(parents=True)
    values = {
        "AF2FT30": (0.890, 0.830, 0.820),
        "AF2CAL3": (0.898, 0.840, 0.830),
    }
    metric_names = (
        "macro_map50_95",
        "bottom3_class_map50_95",
        "worst_class_map50_95",
    )
    for arm, metrics in values.items():
        (reports / f"{arm}_seed42_result.json").write_text(
            json.dumps(
                {
                    "test_images_accessed": False,
                    "metrics": dict(zip(metric_names, metrics)),
                }
            ),
            encoding="utf-8",
        )
    evidence = {
        "format": "coffee_detector.af2r.seed42_recovered_evidence.v1",
        "seed": 42,
        "decision": "FAIL",
        "test_opened": False,
        "values": {
            "AF2": dict(zip(metric_names, (0.882, 0.800, 0.793))),
            "AF2R0": dict(zip(metric_names, (0.895, 0.843, 0.840))),
        },
    }
    evidence_path = tmp_path / "af2r.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    result = run_faruq_v3_af2cal_decision(output, evidence_path)

    assert result["decision"] == "PASS"
    assert result["attribution"] == "CHANNEL_CALIBRATION_SUPPORTED"
    assert result["test_opened"] is False


def test_protocol_and_kaggle_notebook_are_frozen_and_fail_fast():
    protocol = (
        ROOT / "docs/FARUQ_V3_AF2_CHANNEL_CALIBRATION_PROTOCOL.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before training**" in protocol
    assert "`AF2FT30`" in protocol and "`AF2CAL3`" in protocol
    assert "exactly three" in protocol
    assert "Test remains" in protocol

    notebook = json.loads(
        (ROOT / "notebooks/Faruq_V3_AF2_Channel_Calibration_Kaggle.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "ultralytics==8.4.96" in source
    assert "prepare_faruq_v3_kaggle_input" in source
    assert "AF2FT30" in source and "AF2CAL3" in source
    assert "result_path.is_file()" in source
    assert "DOWNLOAD SEBELUM STOP SESSION" in source
