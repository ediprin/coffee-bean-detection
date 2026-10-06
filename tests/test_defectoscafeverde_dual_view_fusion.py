import json
from pathlib import Path

import torch

from coffee_detector.defectos_dual_view import (
    DualViewFusionConfig,
    SymmetricDualViewFuser,
)
from coffee_detector.defectos_dual_view.model import paper_base_logits
from coffee_detector.experiments.run_defectoscafeverde_dual_view_fusion import (
    _cache_reuse_mode,
    _decision,
    _train_fuser,
    run_static_audit,
)


def test_fuser_is_exact_paper_identity_and_swap_invariant_at_initialization():
    torch.manual_seed(4)
    config = DualViewFusionConfig()
    model = SymmetricDualViewFuser(config)
    inputs = torch.randn(5, 2, config.classes)

    output = model(inputs)

    assert torch.equal(output, paper_base_logits(inputs))
    assert torch.equal(output, model(inputs.flip(1)))


def test_active_bounded_residual_changes_scores_without_order_dependence():
    torch.manual_seed(5)
    config = DualViewFusionConfig(residual_limit=2.0)
    model = SymmetricDualViewFuser(config)
    inputs = torch.randn(7, 2, config.classes)
    with torch.no_grad():
        model.residual[-1].bias.copy_(torch.linspace(-0.5, 0.5, config.classes))

    delta = model(inputs) - paper_base_logits(inputs)

    assert not torch.equal(delta, torch.zeros_like(delta))
    assert float(delta.detach().abs().max()) <= config.residual_limit
    assert torch.equal(model(inputs), model(inputs.flip(1)))


def test_static_audit_passes(tmp_path: Path):
    result = run_static_audit(tmp_path / "static.json")

    assert result["decision"] == "PASS"
    assert all(result["gates"].values())
    assert 0 < result["parameters"] < 5000


def test_decision_requires_physical_gain_and_tail_preservation():
    paper = {
        "physical_pair_accuracy": 0.94,
        "macro_class_accuracy": 0.90,
        "bottom3_class_accuracy": 0.75,
        "worst_class_accuracy": 0.70,
    }
    candidate = {
        "physical_pair_accuracy": 0.95,
        "macro_class_accuracy": 0.91,
        "bottom3_class_accuracy": 0.76,
        "worst_class_accuracy": 0.69,
    }
    comparison, decision, next_action = _decision(
        {"PAPER_MAX_CONFIDENCE": paper, "DVF1": candidate},
        {
            "overall_gain": 0.005,
            "macro_floor": 0.0,
            "bottom3_floor": 0.0,
            "worst_drop_limit": 0.02,
        },
    )

    assert decision == "PASS"
    assert next_action == "REVIEW_BEFORE_MULTISEED"
    assert all(comparison["criteria"].values())


def test_completed_fuser_checkpoint_is_reused_without_training(tmp_path: Path):
    torch.manual_seed(7)
    cache = {
        "view_logits": torch.randn(24, 2, 12),
        "labels": torch.arange(24) % 12,
    }
    train_config = {
        "seed": 42,
        "epochs": 1,
        "batch": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "label_smoothing": 0.05,
        "preservation_weight": 0.05,
    }
    checkpoint = tmp_path / "last.pt"
    contract = {"study": "unit"}

    trained, history, first_call = _train_fuser(
        cache,
        DualViewFusionConfig(),
        train_config,
        checkpoint,
        contract,
        torch.device("cpu"),
    )
    reused, reused_history, second_call = _train_fuser(
        cache,
        DualViewFusionConfig(),
        train_config,
        checkpoint,
        contract,
        torch.device("cpu"),
    )

    assert first_call is True
    assert second_call is False
    assert history == reused_history
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "generator_state" in state
    probe = cache["view_logits"][:2]
    assert torch.equal(trained(probe), reused(probe))


def test_only_legacy_train_cache_can_be_upgraded_without_rebuilding():
    expected = {
        "checkpoint": "abc",
        "localization_box_source": "final_top_v1",
    }
    legacy = {"checkpoint": "abc"}

    assert _cache_reuse_mode(expected, expected, "val") == "exact"
    assert _cache_reuse_mode(legacy, expected, "train") == "upgrade_train_only"
    assert _cache_reuse_mode(legacy, expected, "val") == "rebuild"
    assert _cache_reuse_mode({"checkpoint": "wrong"}, expected, "train") == "rebuild"


def test_protocol_precedes_training_and_locks_test():
    text = Path("docs/DEFECTOSCAFEVERDE_DVF1_PROTOCOL_2026-10-06.md").read_text(
        encoding="utf-8"
    )
    assert "before DVF1" in text
    assert "training or validation evaluation" in text
    assert "100 epochs" in text
    assert "Validation fitting" in text
    assert "Test is not extracted or accessed" in text


def test_config_and_protocol_match():
    import yaml

    payload = yaml.safe_load(
        Path("configs/defectoscafeverde/DVF1.yaml").read_text(encoding="utf-8")
    )
    assert payload["code"] == "DVF1"
    assert payload["protocol"] == "defectoscafeverde-dual-view-fusion-seed42-v1"
    assert payload["train"]["epochs"] == 100


def test_colab_compiles_and_keeps_test_locked():
    notebook = json.loads(
        Path("notebooks/DefectosCafeVerde_DVF1_Seed42_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(source, "dvf1_colab", "exec")
    assert "codex/defectoscafeverde-dual-view-fusion" in source
    assert "run_defectoscafeverde_dual_view_fusion" in source
    assert "first in {'train','val'}" in source
    assert "if (DATA/'test').exists()" in source
    assert "--authorize-training" in source
