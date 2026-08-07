from pathlib import Path

import torch
import yaml

from coffee_detector.fsce_cpe import (
    FSCECPEConfig,
    FSCECPEDetectionModel,
    FSCECPEDetectHead,
    aligned_iou_xyxy,
    cpe_supervised_contrastive_loss,
    load_fsce_cpe_detector_weights,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(nc: int = 5, *, iou_threshold: float = 0.7):
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = FSCECPEDetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        cpe=FSCECPEConfig(embedding_dim=16, temperature=0.2, iou_threshold=iou_threshold, loss_weight=0.5),
    ).eval()
    load_fsce_cpe_detector_weights(candidate, source)
    return source, candidate


def test_fsce_cpe_inference_is_exact_native_and_skips_projection():
    source, candidate = _models()
    head = candidate.model[-1]
    assert isinstance(head, FSCECPEDetectHead)
    original = head.cpe_projection.forward

    def forbidden(*args, **kwargs):
        raise AssertionError("CPE projection must not run at inference")

    head.cpe_projection.forward = forbidden
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        source_output = source(image)
        candidate_output = candidate(image)
    head.cpe_projection.forward = original
    assert torch.equal(candidate_output[0], source_output[0])
    assert torch.equal(candidate_output[1]["one2one"]["boxes"], source_output[1]["one2one"]["boxes"])
    assert torch.equal(candidate_output[1]["one2one"]["scores"], source_output[1]["one2one"]["scores"])


def test_cpe_embeddings_exist_only_on_one_to_many_training_branch():
    _, candidate = _models()
    candidate.train()
    output = candidate(torch.randn(2, 3, 128, 128))
    assert "cpe_embeddings" in output["one2many"]
    assert "cpe_embeddings" not in output["one2one"]
    embeddings = output["one2many"]["cpe_embeddings"]
    scores = output["one2many"]["scores"].transpose(1, 2)
    assert embeddings.shape[:2] == scores.shape[:2]
    assert embeddings.shape[2] == 16


def test_cpe_native_boxes_and_scores_unchanged_before_loss():
    source, candidate = _models()
    source.train()
    candidate.train()
    image = torch.randn(1, 3, 128, 128)
    native = source(image)
    cpe = candidate(image)
    for branch in ("one2many", "one2one"):
        assert torch.equal(cpe[branch]["boxes"], native[branch]["boxes"])
        assert torch.equal(cpe[branch]["scores"], native[branch]["scores"])


def test_cpe_loss_rewards_same_class_similarity_and_cross_class_separation():
    labels = torch.tensor([0, 0, 1, 1])
    good = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]], requires_grad=True)
    bad = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1], [-0.9, -0.1]], requires_grad=True)
    good_loss = cpe_supervised_contrastive_loss(good, labels, temperature=0.2)
    bad_loss = cpe_supervised_contrastive_loss(bad, labels, temperature=0.2)
    assert good_loss < bad_loss
    good_loss.backward()
    assert good.grad is not None


def test_cpe_singletons_are_zero_terms_without_nan():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    loss = cpe_supervised_contrastive_loss(embeddings, torch.tensor([0, 1]), temperature=0.2)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_aligned_iou_xyxy_known_values():
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]])
    b = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
    iou = aligned_iou_xyxy(a, b)
    assert torch.allclose(iou, torch.tensor([1.0, 1.0 / 7.0]), atol=1e-7)


def test_predeclared_configs_match_fsce_paper_defaults():
    expected = {
        "CPE0_all_positive.yaml": 0.0,
        "CPE7_iou07.yaml": 0.7,
    }
    for filename, threshold in expected.items():
        payload = yaml.safe_load((ROOT / "configs/fsce_cpe" / filename).read_text(encoding="utf-8"))
        assert payload["cpe"]["embedding_dim"] == 128
        assert payload["cpe"]["temperature"] == 0.2
        assert payload["cpe"]["loss_weight"] == 0.5
        assert payload["cpe"]["iou_threshold"] == threshold
        assert payload["train"]["epochs"] == 50
        assert payload["train"]["imgsz"] == 640
