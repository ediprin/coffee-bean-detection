import torch
import yaml
import json
from types import SimpleNamespace

from coffee_detector.dida_af2 import (
    DIDAAF2Config,
    GTLogits,
    aggregate_positive_logits,
    diversify_appearance,
    match_gt_logits,
    smooth_topk_margin_loss,
    weak_to_strong_consistency,
)


def test_factorial_flags():
    expected = {
        "control": (False, False),
        "dg": (True, False),
        "fg": (False, True),
        "dgfg": (True, True),
    }
    for mode, flags in expected.items():
        config = DIDAAF2Config(mode=mode)
        assert (config.dg_enabled, config.fg_enabled) == flags


def test_style_is_geometry_preserving_and_finite():
    torch.manual_seed(42)
    image = torch.rand(2, 3, 31, 47)
    result = diversify_appearance(image, DIDAAF2Config(mode="dg"))
    assert result.shape == image.shape
    assert torch.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1
    assert not torch.equal(result, image)


def test_positive_logits_are_gt_balanced():
    scores = torch.tensor(
        [[[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [9.0, 9.0]]]
    )
    foreground = torch.tensor([[True, True, True, False]])
    target_gt = torch.tensor([[0, 0, 1, 0]])
    labels = torch.tensor([[0, 0, 1, 0]])
    result = aggregate_positive_logits(scores, foreground, target_gt, labels)
    assert result.keys.tolist() == [[0, 0], [0, 1]]
    assert result.labels.tolist() == [0, 1]
    assert torch.allclose(result.logits, torch.tensor([[2.0, 0.0], [0.0, 2.0]]))


def test_gt_matching_does_not_require_same_anchor_set():
    weak = GTLogits(
        keys=torch.tensor([[0, 0], [0, 1]]),
        labels=torch.tensor([2, 3]),
        logits=torch.randn(2, 5),
    )
    strong = GTLogits(
        keys=torch.tensor([[0, 1], [0, 2], [0, 0]]),
        labels=torch.tensor([3, 4, 2]),
        logits=torch.randn(3, 5),
    )
    left, right = match_gt_logits(weak, strong)
    assert left.shape == right.shape == (2, 5)
    assert torch.equal(left[0], weak.logits[0])
    assert torch.equal(right[0], strong.logits[2])


def test_margin_and_consistency_have_finite_gradients():
    weak_logits = torch.randn(4, 21, requires_grad=True)
    strong_logits = torch.randn(4, 21, requires_grad=True)
    labels = torch.tensor([0, 3, 7, 20])
    keys = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    weak = GTLogits(keys, labels, weak_logits)
    strong = GTLogits(keys, labels, strong_logits)
    fg = smooth_topk_margin_loss(weak_logits, labels, margin=0.2, topk=3)
    dg, matched = weak_to_strong_consistency(weak, strong, temperature=2.0)
    (fg + dg).backward()
    assert matched == 4
    assert torch.isfinite(fg) and torch.isfinite(dg)
    assert torch.isfinite(weak_logits.grad).all()
    assert torch.isfinite(strong_logits.grad).all()


def test_end_to_end_paired_loss_is_finite_and_three_component(tmp_path):
    from pathlib import Path

    from coffee_detector.dida_af2.model import DIDAAF2DetectionModel

    root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (root / "configs/dida_af2/AF2DGFG_yolo26n.yaml").read_text(encoding="utf-8")
    )
    model = DIDAAF2DetectionModel(
        str(root / payload["model"]),
        ch=3,
        nc=21,
        verbose=False,
        afab=payload["afab"],
        dida=payload["dida"],
    ).train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=50)
    image = torch.rand(1, 3, 64, 64)
    batch = {
        "img": image,
        "img_style": diversify_appearance(image, model.dida_config),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[3.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
    }
    loss, items = model.loss(batch)
    assert loss.shape == items.shape == (3,)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_validation_eval_tuple_uses_single_native_view():
    from pathlib import Path

    from coffee_detector.dida_af2.model import DIDAAF2DetectionModel

    root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (root / "configs/dida_af2/AF2FT_yolo26n.yaml").read_text(encoding="utf-8")
    )
    model = DIDAAF2DetectionModel(
        str(root / payload["model"]),
        ch=3,
        nc=21,
        verbose=False,
        afab=payload["afab"],
        dida=payload["dida"],
    )
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=50)
    image = torch.rand(1, 3, 64, 64)
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[3.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
    }
    model.eval()
    predictions = model(image)
    assert isinstance(predictions, tuple)
    assert isinstance(predictions[0], torch.Tensor)
    assert isinstance(predictions[1], dict)
    loss, items = model.loss(batch, predictions)
    assert loss.shape == items.shape == (3,)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()


def test_factorial_decision_requires_joint_to_beat_both_single_factors(tmp_path):
    from coffee_detector.experiments.run_faruq_v3_dida_af2_decision import (
        run_faruq_v3_dida_af2_decision,
    )

    reports = tmp_path / "val_reports"
    reports.mkdir()
    values = {
        "AF2FT": (0.87, 0.78, 0.76),
        "AF2DG": (0.875, 0.79, 0.77),
        "AF2FG": (0.876, 0.795, 0.775),
        "AF2DGFG": (0.885, 0.805, 0.78),
    }
    for arm, metrics in values.items():
        payload = {
            "test_images_accessed": False,
            "metrics": dict(
                zip(
                    (
                        "macro_map50_95",
                        "bottom3_class_map50_95",
                        "worst_class_map50_95",
                    ),
                    metrics,
                )
            ),
        }
        (reports / f"{arm}_seed42_result.json").write_text(json.dumps(payload))
    result = run_faruq_v3_dida_af2_decision(tmp_path)
    assert result["decision"] == "PASS"
    assert result["effects"]["macro_map50_95"]["interaction"] > 0.0
    assert result["test_opened"] is False
