import json
from pathlib import Path

import torch
import yaml

from coffee_detector.defectos_dual_view import (
    SelectiveDualViewFuser,
    SelectiveDualViewFusionConfig,
)
from coffee_detector.defectos_dual_view.model import paper_base_logits
from coffee_detector.experiments.run_defectoscafeverde_dvf2 import (
    _decision,
    _train_fuser,
    run_static_audit,
)


def test_dvf2_is_exact_identity_and_swap_invariant_at_initialization():
    torch.manual_seed(9)
    config = SelectiveDualViewFusionConfig()
    model = SelectiveDualViewFuser(config)
    inputs = torch.randn(7, 2, config.classes)

    output = model(inputs)

    assert torch.equal(output, paper_base_logits(inputs))
    assert torch.equal(output, model(inputs.flip(1)))


def test_dvf2_hard_gate_falls_back_or_activates_expert():
    torch.manual_seed(10)
    config = SelectiveDualViewFusionConfig()
    model = SelectiveDualViewFuser(config)
    inputs = torch.randn(5, 2, config.classes)
    base = paper_base_logits(inputs)
    with torch.no_grad():
        model.expert[-1].bias.fill_(0.2)
        model.gate[-1].weight.zero_()
        model.gate[-1].bias.fill_(-10.0)
        model.eval()
        fallback = model(inputs)
        model.gate[-1].bias.fill_(10.0)
        active = model(inputs)

    assert torch.equal(fallback, base)
    assert not torch.equal(active, base)
    assert torch.equal(active, model(inputs.flip(1)))


def test_dvf2_static_audit_passes(tmp_path: Path):
    result = run_static_audit(tmp_path / "static.json")

    assert result["decision"] == "PASS"
    assert all(result["gates"].values())
    assert 0 < result["parameters"] < 15_000


def test_dvf2_decision_accepts_overall_or_tail_route():
    control = {
        "physical_pair_accuracy": 0.95,
        "macro_class_accuracy": 0.92,
        "bottom3_class_accuracy": 0.80,
        "worst_class_accuracy": 0.75,
    }
    config = {
        "overall_gain_over_dvf1": 0.005,
        "macro_floor_vs_dvf1": 0.0,
        "bottom3_floor_vs_dvf1": 0.0,
        "worst_drop_limit_vs_dvf1": 0.02,
        "tail_gain_over_dvf1": 0.02,
    }
    overall = {**control, "physical_pair_accuracy": 0.956}
    tail = {**control, "bottom3_class_accuracy": 0.825}

    for candidate in (overall, tail):
        comparison, decision, next_action = _decision(
            {"DVF1": control, "DVF2": candidate}, config
        )
        assert decision == "PASS"
        assert next_action == "REVIEW_DVF2_BEFORE_CONFIRMATION"
        assert comparison["overall_route"] or comparison["lower_tail_route"]


def test_dvf2_checkpoint_resumes_without_retraining(tmp_path: Path):
    torch.manual_seed(11)
    cache = {
        "view_logits": torch.randn(48, 2, 12),
        "labels": torch.arange(48) % 12,
    }
    train_config = {
        "seed": 42,
        "epochs": 1,
        "batch": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "label_smoothing": 0.05,
        "gate_loss_weight": 0.25,
        "rescue_loss_weight": 0.5,
        "preservation_weight": 0.10,
    }
    checkpoint = tmp_path / "last.pt"
    contract = {"study": "unit"}

    first, history, trained = _train_fuser(
        cache,
        SelectiveDualViewFusionConfig(),
        train_config,
        checkpoint,
        contract,
        torch.device("cpu"),
    )
    second, reused_history, reused = _train_fuser(
        cache,
        SelectiveDualViewFusionConfig(),
        train_config,
        checkpoint,
        contract,
        torch.device("cpu"),
    )

    assert trained is True
    assert reused is False
    assert history == reused_history
    probe = cache["view_logits"][:4]
    assert torch.equal(first(probe), second(probe))


def test_dvf2_protocol_config_and_notebook_contract():
    protocol = Path("docs/DEFECTOSCAFEVERDE_DVF2_PROTOCOL_2026-10-06.md").read_text(
        encoding="utf-8"
    )
    config = yaml.safe_load(
        Path("configs/defectoscafeverde/DVF2.yaml").read_text(encoding="utf-8")
    )
    notebook = json.loads(
        Path("notebooks/DefectosCafeVerde_DVF2_Seed42_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "before DVF2 training or evaluation" in protocol
    assert "If DVF2 fails, no DVF3" in protocol
    assert config["code"] == "DVF2"
    assert config["train"]["epochs"] == 100
    compile(source, "dvf2_colab", "exec")
    assert "codex/defectoscafeverde-dual-view-fusion" in source
    assert "run_defectoscafeverde_dvf2" in source
    assert "--authorize-training" in source
    assert "test_images_accessed" in source
