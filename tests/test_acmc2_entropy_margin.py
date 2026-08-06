import torch

from coffee_detector.ambiguity_multilevel.model import (
    AmbiguityConditionedFusion,
    AmbiguityMultilevelConfig,
)


def _fusion(mode: str) -> AmbiguityConditionedFusion:
    return AmbiguityConditionedFusion(
        channels=(16, 32, 64),
        num_classes=5,
        config=AmbiguityMultilevelConfig(
            hidden_dim=8,
            ambiguity_mode=mode,
            gate_hidden_dim=4,
            gate_delta_scale=0.5,
        ),
    )


def test_acmc2_zero_initialized_gate_matches_acmc1_entropy() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 4, 4)
    acmc1 = _fusion("entropy")
    acmc2 = _fusion("entropy_margin")
    with torch.inference_mode():
        gate1 = acmc1._ambiguity(logits, 0)
        gate2 = acmc2._ambiguity(logits, 0)
    assert torch.allclose(gate2, gate1, rtol=0.0, atol=1e-7)


def test_acmc2_margin_uncertainty_is_high_when_top_classes_are_close() -> None:
    close = torch.tensor([[[[0.49]], [[0.48]], [[0.01]], [[0.01]], [[0.01]]]])
    clear = torch.tensor([[[[0.90]], [[0.04]], [[0.02]], [[0.02]], [[0.02]]]])
    close_u = AmbiguityConditionedFusion._margin_uncertainty(close)
    clear_u = AmbiguityConditionedFusion._margin_uncertainty(clear)
    assert close_u.item() > clear_u.item()


def test_acmc2_learned_gate_can_depart_from_entropy() -> None:
    torch.manual_seed(1)
    logits = torch.randn(1, 5, 3, 3)
    module = _fusion("entropy_margin")
    gate_module = module.entropy_margin_gates[0]
    assert gate_module is not None
    with torch.no_grad():
        gate_module.network[-1].bias.fill_(0.5)
    probability = logits.detach().softmax(dim=1)
    entropy = module._entropy(probability)
    learned = module._ambiguity(logits, 0)
    assert not torch.allclose(learned, entropy)
    assert bool((learned >= 0.0).all())
    assert bool((learned <= 1.0).all())


def test_acmc2_forward_keeps_shapes_and_zero_correction() -> None:
    torch.manual_seed(2)
    module = _fusion("entropy_margin").eval()
    features = [
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
        torch.randn(1, 64, 2, 2),
    ]
    logits = [
        torch.randn(1, 5, 8, 8),
        torch.randn(1, 5, 4, 4),
        torch.randn(1, 5, 2, 2),
    ]
    with torch.inference_mode():
        corrections = module(features, logits)
    assert [tuple(value.shape) for value in corrections] == [tuple(value.shape) for value in logits]
    assert all(torch.count_nonzero(value).item() == 0 for value in corrections)
