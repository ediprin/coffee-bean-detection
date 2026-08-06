import torch

from coffee_detector.ambiguity_multilevel.model import (
    AmbiguityConditionedFusion,
    AmbiguityMultilevelConfig,
)
from coffee_detector.analysis.acmc1_residual_error_audit_v3 import (
    install_legacy_acmc1_checkpoint_compatibility,
)


def test_missing_entropy_margin_gates_falls_back_to_acmc1_entropy() -> None:
    module = AmbiguityConditionedFusion(
        channels=(8, 16, 32),
        num_classes=5,
        config=AmbiguityMultilevelConfig(hidden_dim=8, ambiguity_mode="entropy"),
    )
    # Simulate an ACMC1 checkpoint serialized before ACMC2 added this attribute.
    del module.entropy_margin_gates
    logits = torch.randn(2, 5, 4, 4)
    probability = logits.detach().softmax(dim=1)
    expected = module._entropy(probability)

    install_legacy_acmc1_checkpoint_compatibility()
    actual = module._ambiguity(logits, 0)

    assert torch.equal(actual, expected)
