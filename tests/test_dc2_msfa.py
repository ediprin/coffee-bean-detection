import json
from pathlib import Path

import numpy as np
import torch

from coffee_detector.dc2_crop.model import build_local_classifier
from coffee_detector.dc2_crop.msfa import DC2MSFAClassifier, MSFAMatchedDataset, trainable_parameter_count
from coffee_detector.experiments.run_faruq_v3_dc2_msfa_screening import (
    GLOBAL_LEVEL,
    PROTOCOL,
    RESOLUTION,
    decide_dc2_msfa,
)


def test_msfa_zero_projection_exactly_reproduces_local_classifier() -> None:
    torch.manual_seed(7)
    local = build_local_classifier(21, imagenet_pretrained=False).eval()
    model = DC2MSFAClassifier(local, global_dim=32).eval()
    images = torch.randn(3, 3, 128, 128)
    global_features = torch.randn(3, 32)
    with torch.inference_mode():
        expected = local(images)
        actual = model(images, global_features)
    assert torch.equal(expected, actual)


def test_only_global_projection_is_trainable() -> None:
    local = build_local_classifier(21, imagenet_pretrained=False)
    model = DC2MSFAClassifier(local, global_dim=16)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable == ["global_projection.weight", "global_projection.bias"]
    assert trainable_parameter_count(model) == sum(
        parameter.numel() for parameter in model.global_projection.parameters()
    )


def test_msfa_gate_requires_material_macro_gain_and_tail_safety() -> None:
    passed = decide_dc2_msfa(
        train_coverage=0.95,
        val_coverage=0.94,
        replay_metrics={"macro_f1": 0.82, "bottom3_f1": 0.70, "worst_f1": 0.62},
        dc2b_metrics={"macro_f1": 0.82, "bottom3_f1": 0.70, "worst_f1": 0.62},
        msfa_metrics={"macro_f1": 0.827, "bottom3_f1": 0.699, "worst_f1": 0.615},
        global_feature_std=0.2,
    )
    assert passed["decision"] == "PASS"
    assert passed["next_action"] == "AUTHORIZE_DC2_END_TO_END_INTEGRATION_SCREENING"
    assert all(passed["criteria"].values())

    failed = decide_dc2_msfa(
        train_coverage=0.95,
        val_coverage=0.94,
        replay_metrics={"macro_f1": 0.82, "bottom3_f1": 0.70, "worst_f1": 0.62},
        dc2b_metrics={"macro_f1": 0.82, "bottom3_f1": 0.70, "worst_f1": 0.62},
        msfa_metrics={"macro_f1": 0.823, "bottom3_f1": 0.69, "worst_f1": 0.60},
        global_feature_std=0.2,
    )
    assert failed["decision"] == "FAIL"
    assert not failed["criteria"]["msfa_macro_gain_at_least_half_point"]


def test_msfa_constants_are_frozen() -> None:
    assert PROTOCOL == "faruq-v3-dc2-msfa-screening-v1"
    assert RESOLUTION == 128
    assert GLOBAL_LEVEL == "P5"


def test_msfa_dataset_rejects_misaligned_global_features() -> None:
    try:
        MSFAMatchedDataset([], np.zeros((1, 8), dtype=np.float32), resolution=128, training=False)
    except ValueError as error:
        assert "tidak sejajar" in str(error)
    else:
        raise AssertionError("Dataset harus menolak jumlah global feature yang tidak sejajar")


def test_dc2_msfa_notebook_is_gated_and_never_requests_test_split() -> None:
    notebook = Path("notebooks/Faruq_V3_DC2_MSFA_Screening_Colab.ipynb")
    assert notebook.is_file()
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "agent/dc2-msfa-screening" in source
    assert "faruq-v3-dc2-predicted-raw-crop-screening-v2" in source
    assert "run_faruq_v3_dc2_msfa_screening" in source
    assert "--authorize-training" in source
    assert "test/" not in source
    assert "split=test" not in source.lower()
