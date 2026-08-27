from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from coffee_detector.af2_complement import (
    AF2ComplementConfig,
    AF2ComplementDetectionModel,
    FrequencySelectionResidual,
    SpaceFrequencySelectionResidual,
    balanced_supervised_contrastive_loss,
    load_af2_complement_weights,
)
from coffee_detector.af2_complement.modules import low_high_split
from coffee_detector.afab import AFABConfig
from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.experiments.run_faruq_v3_af2_complement_decision import (
    run_af2_complement_decision,
)


MODEL_CFG = "configs/coffee_fg/models/yolo26n-p3.yaml"


@pytest.mark.parametrize(
    ("arm", "mode"),
    [
        ("AF2CTRL", "control"),
        ("AF2FS1", "frequency_select"),
        ("AF2SFS1", "space_frequency"),
        ("AF2BHCL1", "bhcl"),
    ],
)
def test_config_arm_mode_contract(arm, mode):
    result = AF2ComplementConfig.from_mapping({"arm": arm, "mode": mode})
    assert result.arm == arm
    assert result.mode == mode


def test_config_rejects_mismatched_arm_mode():
    with pytest.raises(ValueError, match="harus memakai"):
        AF2ComplementConfig.from_mapping({"arm": "AF2FS1", "mode": "bhcl"})


def test_low_high_split_reconstructs_input():
    value = torch.rand(2, 8, 17, 19)
    low, high = low_high_split(value)
    torch.testing.assert_close(low + high, value)


@pytest.mark.parametrize("module_type", [FrequencySelectionResidual, SpaceFrequencySelectionResidual])
def test_feature_adapter_is_identity_then_changes_shared_feature(module_type):
    module = module_type(8)
    value = torch.rand(2, 8, 12, 12, requires_grad=True)
    initial = module(value)
    assert torch.equal(initial, value)
    torch.nn.init.constant_(module.output.weight, 0.05)
    active = module(value)
    assert not torch.equal(active, value)
    active.mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_balanced_contrastive_is_finite_and_differentiable():
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        requires_grad=True,
    )
    loss = balanced_supervised_contrastive_loss(
        embeddings, torch.tensor([0, 0, 1, 1]), temperature=0.1
    )
    assert torch.isfinite(loss)
    assert loss > 0
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_balanced_contrastive_safe_without_positive_pair():
    embeddings = torch.rand(3, 4, requires_grad=True)
    loss = balanced_supervised_contrastive_loss(embeddings, torch.tensor([0, 1, 2]))
    assert loss == 0
    loss.backward()
    assert embeddings.grad is not None


def test_all_arms_start_from_exact_af2_output():
    torch.manual_seed(9)
    afab = AFABConfig(mode="af2")
    source = AFABDetectionModel(MODEL_CFG, nc=21, verbose=False, afab=afab).eval()
    sample = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        expected = source(sample)
    for arm, mode in (
        ("AF2CTRL", "control"),
        ("AF2FS1", "frequency_select"),
        ("AF2SFS1", "space_frequency"),
        ("AF2BHCL1", "bhcl"),
    ):
        model = AF2ComplementDetectionModel(
            MODEL_CFG,
            nc=21,
            verbose=False,
            afab=afab,
            complement=AF2ComplementConfig(arm=arm, mode=mode),
        ).eval()
        load_af2_complement_weights(model, source)
        with torch.inference_mode():
            observed = model(sample)
        expected_tensors = _flatten(expected)
        observed_tensors = _flatten(observed)
        assert len(expected_tensors) == len(observed_tensors)
        assert all(torch.equal(a, b) for a, b in zip(expected_tensors, observed_tensors))


def _flatten(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(_flatten(value[key]))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return []


def test_bhcl_full_loss_backward_smoke():
    model = AF2ComplementDetectionModel(
        MODEL_CFG,
        nc=21,
        verbose=False,
        afab=AFABConfig(mode="af2"),
        complement=AF2ComplementConfig(arm="AF2BHCL1", mode="bhcl"),
    )
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=30)
    model.train()
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        "cls": torch.tensor([[1.0], [1.0], [2.0], [2.0]]),
        "bboxes": torch.tensor(
            [[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]] * 2
        ),
    }
    loss, _items = model.loss(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_decision_accepts_lower_tail_pareto(tmp_path):
    reports = tmp_path / "runs" / "val_reports"
    reports.mkdir(parents=True)
    values = {
        "AF2CTRL": (0.88, 0.80, 0.78),
        "AF2FS1": (0.8795, 0.81, 0.79),
        "AF2SFS1": (0.87, 0.79, 0.77),
        "AF2BHCL1": (0.881, 0.801, 0.781),
    }
    for arm, (macro, bottom, worst) in values.items():
        payload = {
            "metrics": {
                "macro_map50_95": macro,
                "bottom3_class_map50_95": bottom,
                "worst_class_map50_95": worst,
            },
            "test_images_accessed": False,
        }
        (reports / f"{arm}_seed42_result.json").write_text(json.dumps(payload))
    result = run_af2_complement_decision(tmp_path / "runs", tmp_path / "decision.json")
    assert result["decision"] == "PASS"
    assert result["comparisons"]["AF2FS1"]["lower_tail_pareto_gate"] is True
    assert result["winner"] == "AF2FS1"


def test_colab_notebooks_are_separate_quiet_and_test_locked():
    notebook_dir = Path("notebooks")
    names = {
        "AF2CTRL": "Faruq_V3_AF2CTRL_Complement_Colab.ipynb",
        "AF2FS1": "Faruq_V3_AF2FS1_Complement_Colab.ipynb",
        "AF2SFS1": "Faruq_V3_AF2SFS1_Complement_Colab.ipynb",
        "AF2BHCL1": "Faruq_V3_AF2BHCL1_Complement_Colab.ipynb",
    }
    for arm, name in names.items():
        payload = json.loads((notebook_dir / name).read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in payload["cells"]
        )
        assert f"ARM = '{arm}'" in source
        assert "codex/af2-complementary-mechanisms" in source
        assert "--authorize-training" in source
        assert "time.sleep(30)" in source
        assert "epochs % 5 == 0" in source
        assert "exec(" not in source
        assert len(payload["cells"]) >= 5
        assert "test/images" not in source
        assert "--authorize-test" not in source

    audit = (notebook_dir / "Faruq_V3_AF2_Complement_Static_Audit_Colab.ipynb").read_text(
        encoding="utf-8"
    )
    decision = (notebook_dir / "Faruq_V3_AF2_Complement_Decision_Colab.ipynb").read_text(
        encoding="utf-8"
    )
    assert "run_af2_complement_static_audit" in audit
    assert "traceback.print_exc()" in audit
    assert "--authorize-training" not in audit
    assert "run_faruq_v3_af2_complement_decision" in decision
    assert "--authorize-training" not in decision
