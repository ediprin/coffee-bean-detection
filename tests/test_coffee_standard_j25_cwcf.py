import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from coffee_detector.j25_cwcf import (
    CWCFConfig,
    build_cwcf_model,
    build_j25_attribute_matrix,
    chromatic_wavelet_cue,
    haar_decompose,
)


ROOT = Path(__file__).resolve().parents[1]


def test_haar_constant_has_zero_detail_and_finite_gradient():
    value = torch.ones(2, 1, 16, 18, requires_grad=True)
    ll, detail = haar_decompose(value)
    assert ll.shape == (2, 1, 8, 9)
    assert detail.max() < 1e-3
    (ll.mean() + detail.mean()).backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_chromatic_wavelet_cue_is_finite_and_four_channel():
    image = torch.rand(2, 3, 65, 71, requires_grad=True)
    cue = chromatic_wavelet_cue(image)
    assert cue.shape == (2, 4, 65, 71)
    assert torch.isfinite(cue).all()
    assert cue.abs().max() <= 1.0
    cue.mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_black_broken_is_explicit_composition_not_atomic_guess():
    matrix = build_j25_attribute_matrix()
    # J25: 7=black broken, 8=full black, 12=broken.
    assert matrix.shape == (25, 14)
    assert matrix[7, 5:7].tolist() == [1.0, 1.0]
    assert matrix[8, 5:7].tolist() == [1.0, 0.0]
    assert matrix[12, 5:7].tolist() == [0.0, 1.0]


def test_config_matches_safeaug0_fresh_schedule():
    candidate = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/CWCF1.yaml").read_text()
    )
    control = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/SAFEAUG0.yaml").read_text()
    )
    assert candidate["model"] == control["model"]
    assert candidate["train"] == control["train"]
    assert candidate["sampler"] == "none"
    assert CWCFConfig.from_mapping(candidate["cwcf"]).attribute_gain == 0.15


def test_full_model_starts_exact_and_compositional_loss_reaches_adapter():
    from ultralytics.nn.tasks import DetectionModel

    model_yaml = str(ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml")
    torch.manual_seed(42)
    native = DetectionModel(model_yaml, ch=3, nc=25, verbose=False)
    candidate = build_cwcf_model(
        model_yaml, nc=25, source=None, seed=42, config=CWCFConfig(), verbose=False
    )
    image = torch.rand(2, 3, 64, 64)
    native.train()
    candidate.train()
    with torch.no_grad():
        native_raw, candidate_raw = native(image), candidate(image)
    assert torch.equal(native_raw["one2many"]["boxes"], candidate_raw["one2many"]["boxes"])
    assert torch.equal(native_raw["one2many"]["scores"], candidate_raw["one2many"]["scores"])

    candidate.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=50)
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[7.0], [12.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]]),
    }
    loss, _ = candidate(batch)
    loss.sum().backward()
    gradients = [
        parameter.grad
        for parameter in candidate.model[-1].adapters.parameters()
        if parameter.grad is not None
    ]
    assert candidate.last_attribute_loss is not None
    assert torch.isfinite(candidate.last_attribute_loss)
    assert gradients and sum(float(value.abs().sum()) for value in gradients) > 0


def test_notebook_is_fresh_resumable_and_test_locked():
    notebook = json.loads(
        (ROOT / "notebooks/Coffee_Standard_J25_CWCF1_Seed42_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-chromatic-wavelet-composition'" in code
    assert "run_coffee_standard_j25_cwcf" in code
    assert "--authorize-training" in code
    assert "last.pt tersimpan di Drive setiap epoch" in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
