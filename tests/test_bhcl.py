import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from coffee_detector.bhcl import (
    BHCLConfig,
    BHCLDetectHead,
    BHCLDetectionModel,
    BalancedHierarchyPrototypeBank,
    balanced_hierarchical_contrastive_loss,
    balanced_level_loss,
    build_sni21_entity_family_hierarchy,
    hierarchy_level_weights,
    load_bhcl_detector_weights,
    prototype_ema_factor,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def test_sni21_entity_family_is_a_strict_two_level_tree():
    hierarchy = build_sni21_entity_family_hierarchy()
    assert hierarchy.levels == 2
    assert hierarchy.leaf_count == 21
    assert hierarchy.coarse_names == (
        "coffee_bean",
        "dried_coffee_cherry",
        "coffee_husk",
        "parchment",
        "foreign_matter",
    )
    assert len(hierarchy.leaf_to_coarse) == 21
    assert all(0 <= value < hierarchy.coarse_count for value in hierarchy.leaf_to_coarse)
    # Every leaf maps to exactly one parent by construction; no validation confusion is used.
    assert hierarchy.leaf_to_coarse[0] == 0
    assert hierarchy.leaf_to_coarse[11] == 1
    assert hierarchy.leaf_to_coarse[12] == 2
    assert hierarchy.leaf_to_coarse[15] == 3
    assert hierarchy.leaf_to_coarse[18] == 4


def test_hierarchy_penalties_match_eq7_and_favor_fine_level():
    weights = hierarchy_level_weights(2)
    expected_coarse = math.exp(0.5) / (math.exp(0.5) + math.exp(1.0))
    expected_fine = math.exp(1.0) / (math.exp(0.5) + math.exp(1.0))
    assert torch.allclose(weights, torch.tensor([expected_coarse, expected_fine], dtype=torch.float64))
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    assert weights[1] > weights[0]


def test_eq10_momentum_powers_are_literal_for_two_levels():
    assert prototype_ema_factor(0.1, levels=2, level=1) == 0.1
    assert prototype_ema_factor(0.1, levels=2, level=2) == 1.0


def test_zero_initialized_prototypes_follow_eq10_exactly():
    cfg = BHCLConfig(embedding_dim=4, epsilon=0.1)
    bank = BalancedHierarchyPrototypeBank(cfg)
    # leaf 0 and leaf 1 share coffee_bean parent.
    embeddings = F.normalize(
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]), dim=1
    )
    labels = torch.tensor([0, 1])
    bank.update(embeddings, labels)
    expected_coarse = embeddings.mean(dim=0) * 0.1
    assert torch.allclose(bank.coarse_prototypes[0], expected_coarse, atol=1e-7)
    assert torch.allclose(bank.leaf_prototypes[0], embeddings[0], atol=1e-7)
    assert torch.allclose(bank.leaf_prototypes[1], embeddings[1], atol=1e-7)
    # Second identical update: coarse = .9*(.1 mean)+.1*mean = .19 mean.
    bank.update(embeddings, labels)
    assert torch.allclose(bank.coarse_prototypes[0], embeddings.mean(dim=0) * 0.19, atol=1e-7)
    assert torch.allclose(bank.leaf_prototypes[0], embeddings[0], atol=1e-7)


def _naive_balanced_level_loss(z, labels, prototypes, tau):
    values = []
    class_count = prototypes.shape[0]
    for i in range(len(z)):
        denominator = z.new_zeros(())
        for c in range(class_count):
            members = [a for a in range(len(z)) if int(labels[a]) == c]
            divisor = len(members) + 1
            category_sum = z.new_zeros(())
            for a in members:
                if a != i:
                    category_sum = category_sum + torch.exp(torch.dot(z[i], z[a]) / tau)
            category_sum = category_sum + torch.exp(torch.dot(z[i], prototypes[c]) / tau)
            denominator = denominator + category_sum / float(divisor)
        positives = [a for a in range(len(z)) if int(labels[a]) == int(labels[i]) and a != i]
        pair_vectors = [z[a] for a in positives] + [prototypes[int(labels[i])]]
        pair_losses = [
            -torch.log(torch.exp(torch.dot(z[i], p) / tau) / denominator)
            for p in pair_vectors
        ]
        values.append(torch.stack(pair_losses).mean())
    return torch.stack(values).mean()


def test_balanced_level_loss_matches_literal_eq8_reference():
    z = F.normalize(
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.8, 0.2]],
            dtype=torch.float64,
        ),
        dim=1,
    )
    labels = torch.tensor([0, 0, 1, 2])
    prototypes = F.normalize(
        torch.tensor([[1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.1, 1.0]], dtype=torch.float64),
        dim=1,
    )
    observed = balanced_level_loss(z, labels, prototypes, temperature=0.1, anchor_chunk_size=2)
    expected = _naive_balanced_level_loss(z, labels, prototypes, 0.1)
    assert torch.allclose(observed, expected, atol=1e-10, rtol=1e-10)


def test_bhcl_loss_has_finite_gradient_and_updates_both_hierarchy_levels():
    cfg = BHCLConfig(embedding_dim=8, temperature=0.1, epsilon=0.1, anchor_chunk_size=3)
    bank = BalancedHierarchyPrototypeBank(cfg)
    embeddings = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 12, 13, 15, 18])
    loss = balanced_hierarchical_contrastive_loss(embeddings, labels, bank, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
    assert bool(bank.coarse_seen.any())
    assert bool(bank.leaf_seen.any())


def _models():
    from ultralytics.nn.tasks import DetectionModel
    source = DetectionModel(str(MODEL_YAML), nc=21, verbose=False).eval()
    candidate = BHCLDetectionModel(
        str(MODEL_YAML),
        nc=21,
        verbose=False,
        bhcl=BHCLConfig(embedding_dim=32, anchor_chunk_size=32),
    ).eval()
    load_bhcl_detector_weights(candidate, source)
    return source, candidate


def test_bhcl_inference_is_native_and_does_not_execute_projection():
    source, candidate = _models()
    head = candidate.model[-1]
    assert isinstance(head, BHCLDetectHead)
    calls = {"count": 0}
    handle = head.bhcl_projection.register_forward_hook(
        lambda module, inputs, output: calls.__setitem__("count", calls["count"] + 1)
    )
    image = torch.randn(1, 3, 128, 128)
    try:
        with torch.inference_mode():
            native = source(image)
            transferred = candidate(image)
    finally:
        handle.remove()
    assert calls["count"] == 0
    assert torch.allclose(transferred[0], native[0], rtol=0.0, atol=1e-7)
    assert torch.equal(transferred[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"])
    assert torch.equal(transferred[1]["one2one"]["scores"], native[1]["one2one"]["scores"])


def test_training_forward_emits_embeddings_only_for_one2many():
    _, candidate = _models()
    candidate.train()
    output = candidate(torch.randn(1, 3, 128, 128))
    assert "bhcl_embeddings" in output["one2many"]
    assert "bhcl_embeddings" not in output["one2one"]
    assert output["one2many"]["bhcl_embeddings"].shape[-1] == 32


def test_config_freezes_verified_paper_hyperparameters():
    import yaml
    payload = yaml.safe_load((ROOT / "configs/bhcl/BH1_yolo26n_entity_family.yaml").read_text())
    assert payload["bhcl"]["temperature"] == 0.10
    assert payload["bhcl"]["loss_weight"] == 0.60
    assert payload["bhcl"]["epsilon"] == 0.10
    assert payload["bhcl"]["embedding_dim"] == 128
    assert payload["hierarchy"]["coarse_field"] == "entity_family"
    assert payload["hierarchy"]["levels_excluding_root"] == 2


def test_notebook_is_branch_correct_and_val_only():
    notebook = ROOT / "notebooks/Faruq_V3_BHCL_Screening_Colab.ipynb"
    assert notebook.is_file()
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
    assert "agent/bhcl-hierarchical-contrastive-screening" in source
    assert "run_faruq_v3_bhcl_screening" in source
    assert "--authorize-training" in source
    assert "split=test" not in source.lower()
