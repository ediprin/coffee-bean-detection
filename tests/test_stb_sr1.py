import torch

from coffee_detector.stb import STBConfig
from coffee_detector.stb_sr1.model import (
    ClassificationCMCSpatialResidual,
    WindowAttentionResidualBlock,
)


def _config():
    return STBConfig(window_size=4, num_heads=4, mlp_ratio=4.0)


def test_attention_only_block_has_no_mlp_and_preserves_shape():
    block = WindowAttentionResidualBlock(
        16, window_size=4, shift_size=2, num_heads=4
    )
    value = torch.randn(2, 8, 8, 16)
    output = block(value)
    assert output.shape == value.shape
    assert not hasattr(block, "mlp")
    assert not any("mlp" in type(module).__name__.lower() for module in block.modules())


def test_stb_sr1_exact_identity_at_zero_gates():
    torch.manual_seed(42)
    block = ClassificationCMCSpatialResidual(16, _config()).eval()
    value = torch.randn(2, 16, 8, 8)
    with torch.inference_mode():
        output = block(value)
    assert torch.equal(output, value)
    assert float(block.channel_gate) == 0.0
    assert float(block.spatial_gate) == 0.0


def test_channel_and_spatial_gates_each_change_representation():
    torch.manual_seed(42)
    block = ClassificationCMCSpatialResidual(16, _config()).eval()
    value = torch.randn(2, 16, 8, 8)
    with torch.inference_mode():
        zero = block(value)
        block.channel_gate.fill_(0.1)
        channel = block(value)
        block.channel_gate.zero_()
        block.spatial_gate.fill_(0.1)
        spatial = block(value)
    assert not torch.equal(channel, zero)
    assert not torch.equal(spatial, zero)


def test_stb_sr1_gradients_reach_both_gates():
    torch.manual_seed(42)
    block = ClassificationCMCSpatialResidual(16, _config()).train()
    with torch.no_grad():
        block.channel_gate.fill_(0.1)
        block.spatial_gate.fill_(0.1)
    value = torch.randn(2, 16, 8, 8, requires_grad=True)
    block(value).square().mean().backward()
    assert block.channel_gate.grad is not None
    assert block.spatial_gate.grad is not None
    assert value.grad is not None
