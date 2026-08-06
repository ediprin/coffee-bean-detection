from pathlib import Path

import torch

from coffee_detector.ambiguity_multilevel.model import (
    AmbiguityMultilevelConfig,
    AmbiguityMultilevelDetectionModel,
    load_ambiguity_multilevel_detector_weights,
)
from coffee_detector.ambiguity_multilevel.ranking import (
    AmbiguityMultilevelRankingDetectionModel,
    HardCompetitorRankingConfig,
    hard_competitor_softplus_loss,
)
from coffee_detector.experiments.run_faruq_v3_acmc1h_screening import screening_decision


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def test_hard_competitor_loss_uses_strongest_wrong_class() -> None:
    logits = torch.tensor([[2.0, 1.0, -3.0], [0.0, 1.0, 3.0]])
    targets = torch.tensor([0, 2])
    expected = (torch.nn.functional.softplus(torch.tensor(-1.0)) + torch.nn.functional.softplus(torch.tensor(-2.0))) / 2
    actual = hard_competitor_softplus_loss(logits, targets)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0.0)


def test_hcr_config_is_frozen_to_generic_one2one_ranking() -> None:
    config = HardCompetitorRankingConfig.from_mapping(
        {
            "weight": 0.25,
            "branch": "one2one_only",
            "competitor": "strongest_wrong_class",
            "loss": "softplus_pairwise",
        }
    )
    assert config.weight == 0.25
    for bad in (
        {"weight": 0.0},
        {"branch": "one2many"},
        {"competitor": "validation_confusion_pair"},
        {"loss": "hinge"},
    ):
        try:
            HardCompetitorRankingConfig.from_mapping(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Config menyimpang seharusnya ditolak: {bad}")


def test_hcr_adds_no_inference_parameters_or_forward_change_before_learning() -> None:
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=5, verbose=False).eval()
    acmc1 = AmbiguityMultilevelDetectionModel(
        str(MODEL_YAML),
        nc=5,
        verbose=False,
        ambiguity_multilevel=AmbiguityMultilevelConfig(hidden_dim=16),
    ).eval()
    hcr = AmbiguityMultilevelRankingDetectionModel(
        str(MODEL_YAML),
        nc=5,
        verbose=False,
        ambiguity_multilevel=AmbiguityMultilevelConfig(hidden_dim=16),
        hard_competitor_ranking=HardCompetitorRankingConfig(weight=0.25),
    ).eval()
    load_ambiguity_multilevel_detector_weights(acmc1, source)
    load_ambiguity_multilevel_detector_weights(hcr, source)

    assert sum(p.numel() for p in acmc1.parameters()) == sum(p.numel() for p in hcr.parameters())
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        y1 = acmc1(image)[0]
        yh = hcr(image)[0]
    assert torch.allclose(y1, yh, atol=1e-7, rtol=0.0)


def test_screening_gate_requires_nontrivial_macro_and_tail_gain() -> None:
    d0ft = {
        "macro_map50_95": 0.866,
        "bottom3_class_map50_95": 0.766,
        "worst_class_map50_95": 0.730,
    }
    acmc1 = {
        "macro_map50_95": 0.876,
        "bottom3_class_map50_95": 0.791,
        "worst_class_map50_95": 0.763,
    }
    passing = {
        "macro_map50_95": 0.878,
        "bottom3_class_map50_95": 0.796,
        "worst_class_map50_95": 0.762,
    }
    _, _, criteria, decision = screening_decision(d0ft, acmc1, passing)
    assert decision == "PASS"
    assert all(criteria.values())

    tiny = dict(passing)
    tiny["macro_map50_95"] = 0.877
    _, _, _, decision = screening_decision(d0ft, acmc1, tiny)
    assert decision == "FAIL"
