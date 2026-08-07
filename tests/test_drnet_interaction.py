from pathlib import Path

import torch

from coffee_detector.drnet_refinement import (
    DRNetInteractionConfig,
    DRNetInteractionDetectionModel,
    DRNetInteractionDetectHead,
    build_entity_family_mapping,
    load_drnet_interaction_weights,
    verify_fine_logits,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
ONTOLOGY = ROOT / "configs/sni21/structured_ontology_v1.yaml"


def test_entity_family_mapping_is_ontology_only_and_deterministic() -> None:
    names = [
        "biji_normal",
        "biji_hitam",
        "kulit_kopi_ukuran_kecil",
        "kulit_tanduk_ukuran_kecil",
        "kopi_gelondong",
        "tanah_batu_ranting_kecil",
    ]
    mapping, groups, members = build_entity_family_mapping(names, ONTOLOGY)
    assert groups == (
        "coffee_bean",
        "coffee_husk",
        "parchment",
        "dried_coffee_cherry",
        "foreign_matter",
    )
    assert mapping == (0, 0, 1, 2, 3, 4)
    assert members["coffee_bean"] == ["biji_normal", "biji_hitam"]


def test_interaction_verification_suppresses_fine_classes_outside_coarse_group() -> None:
    fine = torch.tensor([[[4.0], [3.0], [8.0], [7.0]]])  # [B=1,C=4,N=1]
    coarse = torch.tensor([[[0.1], [5.0]]])  # predicts group 1
    class_to_group = torch.tensor([0, 0, 1, 1])
    verified, coarse_prediction = verify_fine_logits(
        fine, coarse, class_to_group, floor=-80.0
    )
    assert coarse_prediction.item() == 1
    assert torch.equal(verified[:, 2:], fine[:, 2:])
    assert torch.equal(verified[:, :2], torch.full_like(verified[:, :2], -80.0))


def _models(nc: int = 5):
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = DRNetInteractionDetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        drnet_interaction=DRNetInteractionConfig(),
        class_to_group=(0, 0, 1, 1, 2),
        group_names=("a", "b", "c"),
    ).eval()
    load_drnet_interaction_weights(candidate, source)
    return source, candidate


def test_interaction_head_starts_with_native_fine_training_logits_before_verification() -> None:
    source, candidate = _models()
    head = candidate.model[-1]
    assert isinstance(head, DRNetInteractionDetectHead)
    source.train()
    candidate.train()
    image = torch.randn(1, 3, 128, 128)
    native = source(image)
    interaction = candidate(image)
    for branch in ("one2many", "one2one"):
        assert torch.equal(interaction[branch]["boxes"], native[branch]["boxes"])
        assert torch.equal(interaction[branch]["scores"], native[branch]["scores"])
    assert "dr_coarse_logits" in interaction["one2many"]
    assert "dr_coarse_logits" not in interaction["one2one"]


def test_interaction_inference_applies_coarse_mask_before_native_inference() -> None:
    _, candidate = _models()
    candidate.eval()
    head = candidate.model[-1]
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        _output, predictions = candidate(image)
    one2one = predictions["one2one"]
    assert "dr_coarse_prediction" in one2one
    scores = one2one["scores"]
    coarse_prediction = one2one["dr_coarse_prediction"]
    allowed = head.class_to_group.view(1, -1, 1) == coarse_prediction.unsqueeze(1)
    assert bool(torch.all(scores.masked_select(~allowed) == head.config.verification_floor))
