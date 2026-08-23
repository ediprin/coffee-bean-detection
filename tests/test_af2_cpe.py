from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from coffee_detector.experiments.run_faruq_v3_af2_cpe_decision import decide
from coffee_detector.fsce_cpe.loss import cpe_supervised_contrastive_loss
from coffee_detector.fsce_cpe.model import CPEProjectionHead, FSCECPEConfig

ROOT = Path(__file__).resolve().parents[1]


def _config(arm: str) -> dict:
    return yaml.safe_load((ROOT / f"configs/af2_cpe/{arm}_yolo26n.yaml").read_text(encoding="utf-8"))


def test_matched_configs_preserve_af2_and_cpe0_and_only_change_weight():
    control, candidate = _config("AF2CPE0"), _config("AF2CPE5")
    af2 = yaml.safe_load((ROOT / "configs/afab/AF2_yolo26n_chaotic_amplitude.yaml").read_text())
    cpe0 = yaml.safe_load((ROOT / "configs/fsce_cpe/CPE0_all_positive.yaml").read_text())
    assert control["afab"] == candidate["afab"] == af2["afab"]
    assert control["train"] == candidate["train"] == af2["train"] == cpe0["train"]
    left, right = dict(control["cpe"]), dict(candidate["cpe"])
    assert left.pop("loss_weight") == 0.0
    assert right.pop("loss_weight") == cpe0["cpe"]["loss_weight"] == 0.5
    expected = dict(cpe0["cpe"]); expected.pop("loss_weight")
    assert left == right == expected


def test_projection_gradient_control_zero_candidate_nonzero():
    torch.manual_seed(42)
    head = CPEProjectionHead((4, 4, 4), FSCECPEConfig(embedding_dim=8, iou_threshold=0.0))
    output = head([torch.randn(2, 4, 2, 2) for _ in range(3)]).reshape(-1, 8)
    labels = torch.tensor([0, 0, 1, 1] * 6)
    loss = cpe_supervised_contrastive_loss(output, labels)
    sums = []
    for weight in (0.0, 0.5):
        head.zero_grad(set_to_none=True); (loss * weight).backward(retain_graph=True)
        sums.append(sum(float(p.grad.abs().sum()) for p in head.parameters()))
    assert sums[0] == 0.0 and sums[1] > 0.0


def _result(macro, b3, worst):
    return {"metrics": {"macro_map50_95": macro, "bottom3_class_map50_95": b3,
                        "worst_class_map50_95": worst}}


def test_decision_superiority_route_and_boundaries():
    result = decide(_result(.80, .70, .60), _result(.802, .70, .60))
    assert result["routes"]["superiority"] and result["decision"] == "RETAIN"
    assert result["test_images_accessed"] is False


def test_decision_tail_pareto_route_and_safety_rejection():
    assert decide(_result(.80, .70, .60), _result(.799, .705, .61))["decision"] == "RETAIN"
    rejected = decide(_result(.80, .70, .60), _result(.7979, .72, .63))
    assert rejected["decision"] == "REJECT"


def test_protocol_and_notebook_freeze_scope():
    protocol = (ROOT / "docs/FARUQ_V3_AF2_CPE0_SEED42_PROTOCOL.md").read_text(encoding="utf-8")
    assert "GDS+STB remains blocked" in protocol and "GDSC1" in protocol
    notebook = json.loads((ROOT / "notebooks/Faruq_V3_AF2_CPE0_Seed42_Colab.ipynb").read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "codex/af2-cpe0-seed42" in source
    assert "run_faruq_v3_af2_cpe_static" in source and "AF2CPE0" in source and "AF2CPE5" in source
    assert "--authorize-training" in source and "split='test'" not in source
