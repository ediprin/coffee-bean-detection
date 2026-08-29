from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from coffee_detector.afab import AFABConfig
from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.af2_spds import (
    AF2SPDSConfig,
    AF2SPDSDetectionModel,
    AuxiliaryReconstructionDetectHead,
    load_af2_spds_weights,
    multilevel_reconstruction_loss,
)
from coffee_detector.experiments.run_faruq_v3_af2_spds_decision import (
    run_af2_spds_decision,
)
from coffee_detector.af2_spds.audit import CUDA_OUTPUT_ATOL, max_abs_difference


MODEL_CFG = "configs/coffee_fg/models/yolo26n-p3.yaml"


def _flatten(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(_flatten(value[key]))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return []


@pytest.mark.parametrize(
    ("arm", "target"),
    [("AF2BASE", "none"), ("AF2RGBDS", "rgb"), ("AF2SPDS", "af2_signal")],
)
def test_config_contract(arm, target):
    config = AF2SPDSConfig.from_mapping({"arm": arm, "target": target})
    assert config.target == target


def test_config_rejects_mismatched_arm_target():
    with pytest.raises(ValueError, match="harus memakai"):
        AF2SPDSConfig.from_mapping({"arm": "AF2SPDS", "target": "rgb"})


def test_static_audit_uses_explicit_cuda_numerical_bound():
    reference = {"scores": torch.tensor([1.0]), "boxes": torch.tensor([2.0])}
    equivalent = {
        "scores": torch.tensor([1.0 + CUDA_OUTPUT_ATOL / 2]),
        "boxes": torch.tensor([2.0]),
    }
    assert not torch.equal(reference["scores"], equivalent["scores"])
    assert max_abs_difference(reference, equivalent) <= CUDA_OUTPUT_ATOL


def test_multilevel_reconstruction_is_finite_and_differentiable():
    predictions = [
        torch.rand(2, 3, side, side, requires_grad=True) for side in (16, 8, 4)
    ]
    target = torch.rand(2, 3, 64, 64)
    loss = multilevel_reconstruction_loss(predictions, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in predictions)


def test_auxiliary_head_is_read_only_for_detector_features():
    source = AFABDetectionModel(
        MODEL_CFG, nc=21, verbose=False, afab=AFABConfig(mode="af2")
    )
    head = AuxiliaryReconstructionDetectHead(
        source.model[-1], AF2SPDSConfig(arm="AF2SPDS", target="af2_signal")
    ).train()
    channels = [next(m.in_channels for m in branch.modules() if isinstance(m, torch.nn.Conv2d)) for branch in head.base_head.cv2]
    features = [torch.rand(2, channel, side, side) for channel, side in zip(channels, (16, 8, 4))]
    identities = [value.clone() for value in features]
    head(features)
    assert all(torch.equal(left, right) for left, right in zip(features, identities))
    assert [value.shape for value in head.last_auxiliary_predictions] == [
        (2, 3, 16, 16), (2, 3, 8, 8), (2, 3, 4, 4)
    ]


def test_all_arms_preserve_initial_af2_detector_output_exactly():
    torch.manual_seed(17)
    afab = AFABConfig(mode="af2")
    source = AFABDetectionModel(MODEL_CFG, nc=21, verbose=False, afab=afab).eval()
    sample = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        expected = source(sample)
    for arm, target in (("AF2BASE", "none"), ("AF2RGBDS", "rgb"), ("AF2SPDS", "af2_signal")):
        model = AF2SPDSDetectionModel(
            MODEL_CFG,
            nc=21,
            verbose=False,
            afab=afab,
            spds=AF2SPDSConfig(arm=arm, target=target),
        ).eval()
        load_af2_spds_weights(model, source)
        with torch.inference_mode():
            observed = model(sample)
        first, second = _flatten(expected), _flatten(observed)
        assert len(first) == len(second)
        assert all(torch.equal(a, b) for a, b in zip(first, second))


@pytest.mark.parametrize(
    ("arm", "target"),
    [("AF2BASE", "none"), ("AF2RGBDS", "rgb"), ("AF2SPDS", "af2_signal")],
)
def test_full_loss_backward_smoke(arm, target):
    model = AF2SPDSDetectionModel(
        MODEL_CFG,
        nc=21,
        verbose=False,
        afab=AFABConfig(mode="af2"),
        spds=AF2SPDSConfig(arm=arm, target=target),
    )
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=30)
    model.train()
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "bboxes": torch.tensor([[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]]),
    }
    loss, items = model.loss(batch)
    assert loss.shape == items.shape == (3,)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    head = model.model[-1]
    assert isinstance(head, AuxiliaryReconstructionDetectHead)
    gradients = [parameter.grad for parameter in head.decoders.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    if target == "none":
        assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)
    else:
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_validation_loss_is_native_when_auxiliary_decoders_are_inactive():
    model = AF2SPDSDetectionModel(
        MODEL_CFG,
        nc=21,
        verbose=False,
        afab=AFABConfig(mode="af2"),
        spds=AF2SPDSConfig(arm="AF2SPDS", target="af2_signal"),
    )
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=30)
    model.eval()
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "bboxes": torch.tensor([[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]]),
    }
    with torch.no_grad():
        predictions = model(batch["img"])
        loss, items = model.loss(batch, predictions)
    assert loss.shape == items.shape == (3,)
    assert torch.isfinite(loss).all()
    assert model.model[-1].last_auxiliary_predictions is None


def test_stripped_model_preserves_output_and_native_state_schema():
    from coffee_detector.af2_spds.model import strip_auxiliary_head

    source = AFABDetectionModel(
        MODEL_CFG, nc=21, verbose=False, afab=AFABConfig(mode="af2")
    ).eval()
    candidate = AF2SPDSDetectionModel(
        MODEL_CFG,
        nc=21,
        verbose=False,
        afab=AFABConfig(mode="af2"),
        spds=AF2SPDSConfig(arm="AF2SPDS", target="af2_signal"),
    ).eval()
    load_af2_spds_weights(candidate, source)
    sample = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        before = candidate(sample)
    strip_auxiliary_head(candidate)
    with torch.inference_mode():
        after = candidate(sample)
    assert all(torch.equal(a, b) for a, b in zip(_flatten(before), _flatten(after)))
    assert candidate.state_dict().keys() == source.state_dict().keys()


def test_decision_requires_signal_specific_gain(tmp_path):
    reports = tmp_path / "runs" / "val_reports"
    reports.mkdir(parents=True)
    values = {
        "AF2BASE": (0.890, 0.838, 0.835),
        "AF2RGBDS": (0.892, 0.840, 0.836),
        "AF2SPDS": (0.898, 0.846, 0.842),
    }
    for arm, (macro, bottom, worst) in values.items():
        payload = {
            "metrics": {
                "macro_map50_95": macro,
                "bottom3_class_map50_95": bottom,
                "worst_class_map50_95": worst,
                "classes_without_ground_truth": [],
                "map50_95_by_class": {
                    f"class_{index:02d}": macro + index / 10000.0
                    for index in range(21)
                },
            },
            "test_images_accessed": False,
        }
        (reports / f"{arm}_seed42_result.json").write_text(json.dumps(payload))
    result = run_af2_spds_decision(tmp_path / "runs", tmp_path / "decision.json")
    assert result["decision"] == "PASS"
    assert result["criteria"]["cue_specific_evidence"] is True
    assert len(result["per_class"]) == 21
    assert result["class_summary"]["spds_improved_vs_base"] == 21


def test_colab_notebooks_are_separate_quiet_resumable_and_test_locked():
    names = {
        "AF2BASE": "Faruq_V3_AF2BASE_SPDS_Colab.ipynb",
        "AF2RGBDS": "Faruq_V3_AF2RGBDS_Colab.ipynb",
        "AF2SPDS": "Faruq_V3_AF2SPDS_Colab.ipynb",
    }
    for arm, name in names.items():
        path = Path("notebooks") / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        for index, cell in enumerate(payload["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell["source"]), f"{path}:{index}", "exec")
        assert f"ARM='{arm}'" in source
        assert "codex/af2-signal-preservation-deep-supervision" in source
        assert "--authorize-training" in source
        assert "last.pt" not in source or "RESUME" in source
        assert "epochs%5==0" in source
        assert "test/images" not in source
        assert "--authorize-test" not in source

    audit = json.loads(
        (Path("notebooks") / "Faruq_V3_AF2_SPDS_Static_Audit_Colab.ipynb").read_text()
    )
    decision = json.loads(
        (Path("notebooks") / "Faruq_V3_AF2_SPDS_Decision_Colab.ipynb").read_text()
    )
    for payload in (audit, decision):
        for index, cell in enumerate(payload["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell["source"]), f"notebook:{index}", "exec")
        source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        assert "--authorize-training" not in source
        assert "--authorize-test" not in source
