import math
from pathlib import Path

import torch

from coffee_detector.hiervip import (
    HierarchySpec,
    HierarchicalPrototypeTree,
    HierVIPConfig,
    HierVIPDetectionModel,
    build_sni_hierarchy,
    load_hiervip_detector_weights,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
ONTOLOGY = ROOT / "configs/sni21/structured_ontology_v1.yaml"


def test_sni_hierarchy_is_tree_and_gamma_matches_expertdet() -> None:
    names = [
        "biji_normal",
        "biji_hitam",
        "biji_hitam_sebagian",
        "kulit_kopi_ukuran_kecil",
        "kulit_tanduk_ukuran_kecil",
        "tanah_batu_ranting_kecil",
    ]
    hierarchy = build_sni_hierarchy(names, ONTOLOGY)
    assert hierarchy.level_names == (
        "fine_class",
        "primary_condition",
        "entity_family",
        "root",
    )
    assert hierarchy.gamma == (3.0, 2.0, 1.0, 0.0)
    assert hierarchy.level_categories[0] == tuple(names)
    assert hierarchy.level_categories[-1] == ("coffee_quality_sample",)
    # black and partial-black share the same primary condition and entity family.
    assert hierarchy.class_to_level[1][1] == hierarchy.class_to_level[1][2]
    assert hierarchy.class_to_level[2][0] == hierarchy.class_to_level[2][1]


def _simple_hierarchy() -> HierarchySpec:
    return HierarchySpec(
        class_names=("a", "b"),
        level_names=("fine", "family", "root"),
        level_categories=(("a", "b"), ("bean",), ("root",)),
        class_to_level=((0, 1), (0, 0), (0, 0)),
        gamma=(2.0, 1.0, 0.0),
    )


def test_hiervip_first_activation_and_adaptive_momentum_follow_eq4_to_eq7() -> None:
    config = HierVIPConfig(embedding_dim=2)
    tree = HierarchicalPrototypeTree(_simple_hierarchy(), config)
    tree.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
    assert bool(tree.active(0)[0])
    assert torch.allclose(tree.prototypes(0)[0], torch.tensor([1.0, 0.0]))

    stats = tree.update(torch.tensor([[0.0, 1.0]]), torch.tensor([0]))
    # d=1, so w=clip(0.8 - 0.2*1, 0.5, 0.8)=0.6.
    expected = torch.tensor([0.6, 0.4])
    expected = expected / expected.norm()
    assert torch.allclose(tree.prototypes(0)[0], expected, atol=1e-6)
    assert 0.5 <= stats["mean_momentum"] <= 0.8


def test_hiervip_hsc_matches_weighted_same_level_cross_entropy() -> None:
    config = HierVIPConfig(embedding_dim=2, temperature=0.2)
    tree = HierarchicalPrototypeTree(_simple_hierarchy(), config)
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    tree.update(embeddings.detach(), labels)
    loss = tree.loss(embeddings, labels)
    # Fine level logits are [5,0] / [0,5]; family level has one category => CE=0.
    fine_ce = math.log1p(math.exp(-5.0))
    expected = (2.0 * fine_ce) / 3.0
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)
    loss.backward()
    assert embeddings.grad is not None
    assert not any(parameter.requires_grad for parameter in tree.parameters())


def _models(nc: int = 5):
    from ultralytics.nn.tasks import DetectionModel

    hierarchy = HierarchySpec(
        class_names=tuple(f"c{i}" for i in range(nc)),
        level_names=("fine", "family", "root"),
        level_categories=(tuple(f"c{i}" for i in range(nc)), ("a", "b"), ("root",)),
        class_to_level=(tuple(range(nc)), tuple(i % 2 for i in range(nc)), tuple(0 for _ in range(nc))),
        gamma=(2.0, 1.0, 0.0),
    )
    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = HierVIPDetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        hiervip=HierVIPConfig(embedding_dim=16),
        hierarchy=hierarchy,
    ).eval()
    load_hiervip_detector_weights(candidate, source)
    return source, candidate


def test_hiervip_inference_is_exact_native_d0_and_skips_auxiliary() -> None:
    source, candidate = _models()
    head = candidate.model[-1]
    original = head.hiervip.forward

    def forbidden(*args, **kwargs):
        raise AssertionError("HierVIP auxiliary branch must not run at inference")

    head.hiervip.forward = forbidden
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        native = source(image)
        transferred = candidate(image)
    head.hiervip.forward = original
    assert torch.equal(transferred[0], native[0])
    assert torch.equal(
        transferred[1]["one2one"]["scores"], native[1]["one2one"]["scores"]
    )


def test_hiervip_training_keeps_native_detection_outputs_before_auxiliary_loss() -> None:
    source, candidate = _models()
    source.train()
    candidate.train()
    image = torch.randn(1, 3, 128, 128)
    native = source(image)
    transferred = candidate(image)
    assert "hiervip_embeddings" in transferred["one2many"]
    assert "hiervip_embeddings" not in transferred["one2one"]
    for branch in ("one2many", "one2one"):
        assert torch.equal(transferred[branch]["boxes"], native[branch]["boxes"])
        assert torch.equal(transferred[branch]["scores"], native[branch]["scores"])
