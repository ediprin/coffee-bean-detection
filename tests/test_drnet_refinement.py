from pathlib import Path

import torch

from coffee_detector.drnet_refinement import (
    DRNetRefinementConfig,
    DRNetRefinementDetectionModel,
    DRNetRefinementDetectHead,
    DualRefinement,
    confusion_minimized_positive_loss,
    load_drnet_refinement_weights,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(nc: int = 5, *, use_cml: bool = False):
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = DRNetRefinementDetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        drnet_refinement=DRNetRefinementConfig(use_cml=use_cml),
    ).eval()
    load_drnet_refinement_weights(candidate, source)
    return source, candidate


def test_dual_refinement_starts_as_identity() -> None:
    module = DualRefinement(8).eval()
    value = torch.randn(2, 8, 5, 7)
    with torch.inference_mode():
        output = module(value)
    assert torch.allclose(output, value, rtol=0.0, atol=1e-7)


def test_drf1_starts_at_native_d0_and_preserves_boxes() -> None:
    source, candidate = _models()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        source_output = source(image)
        candidate_output = candidate(image)
    head = candidate.model[-1]
    assert isinstance(head, DRNetRefinementDetectHead)
    assert torch.allclose(candidate_output[0], source_output[0], rtol=0.0, atol=1e-7)
    assert torch.equal(
        candidate_output[1]["one2one"]["boxes"],
        source_output[1]["one2one"]["boxes"],
    )
    assert torch.equal(
        candidate_output[1]["one2one"]["scores"],
        source_output[1]["one2one"]["scores"],
    )


def test_active_fine_grained_branch_changes_scores_only() -> None:
    _, candidate = _models()
    candidate.eval()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        before = candidate(image)
    head = candidate.model[-1]
    with torch.no_grad():
        head.fine_grained.classifiers[0].bias.fill_(0.25)
    with torch.inference_mode():
        after = candidate(image)
    assert torch.equal(
        before[1]["one2one"]["boxes"], after[1]["one2one"]["boxes"]
    )
    assert not torch.equal(
        before[1]["one2one"]["scores"], after[1]["one2one"]["scores"]
    )


def test_training_exposes_cml_logits_only_when_enabled() -> None:
    _, drf = _models(use_cml=False)
    drf.train()
    output = drf(torch.randn(1, 3, 128, 128))
    assert "dr_fine_logits" not in output["one2many"]

    _, drc = _models(use_cml=True)
    drc.train()
    output = drc(torch.randn(1, 3, 128, 128))
    assert "dr_fine_logits" in output["one2many"]
    assert output["one2many"]["dr_fine_logits"].shape[-1] == 5
    assert "dr_fine_logits" not in output["one2one"]


def test_confusion_minimized_loss_weights_wrong_samples_more() -> None:
    labels = torch.tensor([0, 0])
    # Sample 0: correct class clearly wins. Sample 1: class 1 wins instead.
    logits = torch.tensor([[4.0, -2.0, -2.0], [-2.0, 4.0, -2.0]])
    loss, details = confusion_minimized_positive_loss(
        logits, labels, lambda1=0.4, lambda2=0.05
    )
    assert torch.isfinite(loss)
    probability = logits.sigmoid()
    s_easy = probability[0, 0] - probability[0, 1:].max()
    s_wrong = probability[1, 0] - probability[1, 1:].max()
    w_easy = torch.exp(-10.0 * (s_easy - 0.4))
    w_wrong = -0.05 * torch.log(s_wrong + 1.0) + 1.0
    assert w_wrong > 1.0
    assert w_easy < 1.0
    assert details["hard_fraction"] == 0.5
