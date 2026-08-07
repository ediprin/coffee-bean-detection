from pathlib import Path

import torch

from coffee_detector.apcl import (
    APCLConfig,
    APCLDetectionModel,
    APCLDetectHead,
    AdaptivePrototypeContrast,
    load_apcl_detector_weights,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(nc: int = 5):
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = APCLDetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        apcl=APCLConfig(embedding_dim=16),
    ).eval()
    load_apcl_detector_weights(candidate, source)
    return source, candidate


def test_apcl_inference_is_exact_native_d0_and_skips_projection() -> None:
    source, candidate = _models()
    head = candidate.model[-1]
    assert isinstance(head, APCLDetectHead)

    original_forward = head.apcl.forward
    def forbidden(*args, **kwargs):
        raise AssertionError("APCL projection must not run in eval/inference")
    head.apcl.forward = forbidden
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        source_output = source(image)
        candidate_output = candidate(image)
    head.apcl.forward = original_forward

    assert torch.equal(candidate_output[0], source_output[0])
    assert torch.equal(
        candidate_output[1]["one2one"]["boxes"],
        source_output[1]["one2one"]["boxes"],
    )
    assert torch.equal(
        candidate_output[1]["one2one"]["scores"],
        source_output[1]["one2one"]["scores"],
    )


def test_apcl_embeddings_exist_only_on_one_to_many_training_branch() -> None:
    _, candidate = _models()
    candidate.train()
    output = candidate(torch.randn(2, 3, 128, 128))
    assert "apcl_embeddings" in output["one2many"]
    assert "apcl_embeddings" not in output["one2one"]
    embeddings = output["one2many"]["apcl_embeddings"]
    scores = output["one2many"]["scores"].transpose(1, 2)
    assert embeddings.shape[:2] == scores.shape[:2]
    assert embeddings.shape[2] == 16


def test_apcl_formula_penalizes_only_other_class_positive_cosine() -> None:
    module = AdaptivePrototypeContrast(num_classes=3, embedding_dim=2, eta=0.4)
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    loss = module.update_and_loss(embeddings, labels)
    expected = (2.0 ** 0.5) / 6.0
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)
    loss.backward()
    assert embeddings.grad is not None
    assert bool(module.prototype_seen[0])
    assert bool(module.prototype_seen[1])
    assert not bool(module.prototype_seen[2])


def test_apcl_ema_updates_prototype_without_own_class_attraction_term() -> None:
    module = AdaptivePrototypeContrast(num_classes=2, embedding_dim=2, eta=0.4)
    first = torch.tensor([[1.0, 0.0]], requires_grad=True)
    loss_first = module.update_and_loss(first, torch.tensor([0]))
    assert loss_first.item() == 0.0
    assert torch.equal(module.prototypes[0], torch.tensor([1.0, 0.0]))

    second = torch.tensor([[0.0, 1.0]], requires_grad=True)
    module.update_and_loss(second, torch.tensor([0]))
    assert torch.allclose(module.prototypes[0], torch.tensor([0.6, 0.4]), atol=1e-7)


def test_apcl_native_training_scores_and_boxes_are_unchanged_before_loss() -> None:
    source, candidate = _models()
    source.train()
    candidate.train()
    image = torch.randn(1, 3, 128, 128)
    source_output = source(image)
    candidate_output = candidate(image)
    for branch in ("one2many", "one2one"):
        assert torch.equal(candidate_output[branch]["boxes"], source_output[branch]["boxes"])
        assert torch.equal(candidate_output[branch]["scores"], source_output[branch]["scores"])
