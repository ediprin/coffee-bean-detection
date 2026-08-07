import json
from pathlib import Path

import torch
import torch.nn.functional as F

from coffee_detector.bhcl import (
    BHCLConfig,
    BalancedHierarchyPrototypeBank,
    hierarchical_contrastive_loss,
)

ROOT = Path(__file__).resolve().parents[1]


def _naive_hcl(z, leaf_labels, bank, tau):
    hierarchy = bank.hierarchy
    coarse = hierarchy.coarse_labels(leaf_labels)
    levels = (coarse, leaf_labels)
    total = z.new_zeros(())
    for i in range(len(z)):
        denominator = sum(
            torch.exp(torch.dot(z[i], z[a]) / tau)
            for a in range(len(z)) if a != i
        )
        for level_index, labels in enumerate(levels):
            positives = [a for a in range(len(z)) if a != i and int(labels[a]) == int(labels[i])]
            if not positives:
                continue
            pair_losses = [
                -torch.log(torch.exp(torch.dot(z[i], z[p]) / tau) / denominator)
                for p in positives
            ]
            total = total + bank.level_weights[level_index].to(z) * torch.stack(pair_losses).mean()
    return total / float(len(z))


def test_hcl_matches_literal_eq6_eq7_reference_without_prototypes():
    cfg = BHCLConfig(embedding_dim=6, temperature=0.1, loss_weight=0.6, variant="hcl", anchor_chunk_size=2)
    bank = BalancedHierarchyPrototypeBank(cfg)
    leaf = torch.tensor([0, 0, 1, 1, 12, 12])
    raw = torch.randn(6, 6, dtype=torch.float64)
    z = F.normalize(raw, dim=1)
    observed = hierarchical_contrastive_loss(raw, leaf, bank, cfg)
    expected = _naive_hcl(z, leaf, bank, 0.1)
    assert torch.allclose(observed, expected, atol=1e-10, rtol=1e-10)
    assert not bool(bank.coarse_seen.any())
    assert not bool(bank.leaf_seen.any())


def test_hcl_has_gradient_but_does_not_update_prototype_state():
    cfg = BHCLConfig(embedding_dim=8, variant="hcl")
    bank = BalancedHierarchyPrototypeBank(cfg)
    embeddings = torch.randn(8, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 12, 12, 15, 15])
    loss = hierarchical_contrastive_loss(embeddings, labels, bank, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
    assert torch.count_nonzero(bank.coarse_prototypes) == 0
    assert torch.count_nonzero(bank.leaf_prototypes) == 0


def test_hcl_and_bhcl_configs_match_except_balancing_variant():
    import yaml
    hcl = yaml.safe_load((ROOT / "configs/bhcl/HCL1_yolo26n_entity_family.yaml").read_text())
    bhcl = yaml.safe_load((ROOT / "configs/bhcl/BH1_yolo26n_entity_family.yaml").read_text())
    assert hcl["bhcl"]["variant"] == "hcl"
    assert bhcl["bhcl"]["variant"] == "bhcl"
    for key in ("embedding_dim", "temperature", "loss_weight", "epsilon", "anchor_chunk_size"):
        assert hcl["bhcl"][key] == bhcl["bhcl"][key]
    assert hcl["hierarchy"] == bhcl["hierarchy"]
    assert hcl["train"] == bhcl["train"]


def test_hcl_bhcl_notebook_is_val_only():
    notebook = ROOT / "notebooks/Faruq_V3_HCL_BHCL_Screening_Colab.ipynb"
    assert notebook.is_file()
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
    assert "agent/bhcl-hierarchical-contrastive-screening" in source
    assert "run_faruq_v3_hcl_bhcl_screening" in source
    assert "--authorize-training" in source
    assert "split=test" not in source.lower()
