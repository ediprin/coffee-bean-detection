import math
from pathlib import Path

import torch

from coffee_detector.pcl import (
    LearnedPrototypeContrast,
    PCLConfig,
    PCLDetectionModel,
    PCLDetectHead,
    load_pcl_detector_weights,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(nc: int = 5):
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = PCLDetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        pcl=PCLConfig(embedding_dim=16),
    ).eval()
    load_pcl_detector_weights(candidate, source)
    return source, candidate


def test_pcl_inference_is_exact_native_d0_and_skips_projection() -> None:
    source, candidate = _models()
    head = candidate.model[-1]
    assert isinstance(head, PCLDetectHead)

    original_forward = head.pcl.forward
    def forbidden(*args, **kwargs):
        raise AssertionError("PCL projection must not run in eval/inference")
    head.pcl.forward = forbidden
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        source_output = source(image)
        candidate_output = candidate(image)
    head.pcl.forward = original_forward

    assert torch.equal(candidate_output[0], source_output[0])
    assert torch.equal(
        candidate_output[1]["one2one"]["boxes"],
        source_output[1]["one2one"]["boxes"],
    )
    assert torch.equal(
        candidate_output[1]["one2one"]["scores"],
        source_output[1]["one2one"]["scores"],
    )


def test_pcl_embeddings_exist_only_on_one_to_many_training_branch() -> None:
    _, candidate = _models()
    candidate.train()
    output = candidate(torch.randn(2, 3, 128, 128))
    assert "pcl_embeddings" in output["one2many"]
    assert "pcl_embeddings" not in output["one2one"]
    embeddings = output["one2many"]["pcl_embeddings"]
    scores = output["one2many"]["scores"].transpose(1, 2)
    assert embeddings.shape[:2] == scores.shape[:2]
    assert embeddings.shape[2] == 16


def test_pcl_protocl_eq3_matches_simple_two_class_case() -> None:
    module = LearnedPrototypeContrast(
        num_classes=2,
        embedding_dim=2,
        temperature=1.0,
        prototype_init_std=1.0,
    )
    with torch.no_grad():
        module.prototypes.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    embeddings = torch.tensor([[1.0, 0.0]], requires_grad=True)
    loss = module.loss(embeddings, torch.tensor([0]))
    expected = math.log1p(math.exp(-1.0)) + 0.5 * math.log(2.0)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)
    loss.backward()
    assert embeddings.grad is not None
    assert module.prototypes.grad is not None


def test_pcl_prototypes_are_learnable_parameters_not_ema_buffers() -> None:
    module = LearnedPrototypeContrast(
        num_classes=3,
        embedding_dim=4,
        temperature=1.0 / 32.0,
        prototype_init_std=1.0,
    )
    parameters = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    assert "prototypes" in parameters
    assert "prototypes" not in buffers
    before = module.prototypes.detach().clone()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    embedding = torch.randn(3, 4, requires_grad=True)
    loss = module.loss(embedding, torch.tensor([0, 1, 2]))
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, module.prototypes.detach())


def test_pcl_native_training_scores_and_boxes_are_unchanged_before_loss() -> None:
    source, candidate = _models()
    source.train()
    candidate.train()
    image = torch.randn(1, 3, 128, 128)
    source_output = source(image)
    candidate_output = candidate(image)
    for branch in ("one2many", "one2one"):
        assert torch.equal(candidate_output[branch]["boxes"], source_output[branch]["boxes"])
        assert torch.equal(candidate_output[branch]["scores"], source_output[branch]["scores"])
