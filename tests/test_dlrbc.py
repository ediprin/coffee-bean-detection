import torch

from coffee_detector.dlrbc.model import (
    DLRBCConfig,
    DLRBCDetectionModel,
    DLRBCDetectHead,
    LowRankClassResidual,
)


MODEL = "configs/coffee_fg/models/yolo26n-p3.yaml"


def test_config_reduces_actual_yolo26n_class_tower_width():
    config = DLRBCConfig()
    assert config.rank == 8
    assert config.projection_channels(64) == 32
    assert config.rank <= config.projection_channels(64) < 64


def test_linear_and_quadratic_have_identical_state_schema_and_initial_values():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        linear = LowRankClassResidual(64, 21, DLRBCConfig(mode="linear"))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        quadratic = LowRankClassResidual(64, 21, DLRBCConfig(mode="quadratic"))
    assert list(linear.state_dict()) == list(quadratic.state_dict())
    assert all(
        torch.equal(linear.state_dict()[key], quadratic.state_dict()[key])
        for key in linear.state_dict()
    )
    assert sum(p.numel() for p in linear.parameters()) == sum(
        p.numel() for p in quadratic.parameters()
    )


def test_quadratic_formula_matches_signed_factor_energy():
    module = LowRankClassResidual(
        16,
        2,
        DLRBCConfig(
            mode="quadratic",
            rank=4,
            projection_ratio=0.5,
            minimum_projection=4,
            residual_scale=1.0,
            signed_sqrt=False,
        ),
    )
    with torch.no_grad():
        module.projection.weight.zero_()
        for index in range(module.projection_channels):
            module.projection.weight[index, index, 0, 0] = 1.0
        module.factors.weight.zero_()
        module.factors.weight[0, 0, 0, 0] = 1.0
        module.factors.weight[1, 1, 0, 0] = 1.0
        module.factors.weight[2, 2, 0, 0] = 1.0
        module.factors.weight[3, 3, 0, 0] = 1.0
    value = torch.zeros(1, 16, 1, 1)
    value[0, :4, 0, 0] = torch.tensor([2.0, 3.0, 1.0, 2.0])
    expected = torch.tensor((4.0 + 9.0 - 1.0 - 4.0) / (2.0**0.5))
    assert torch.allclose(module(value)[0, 0, 0, 0], expected)
    assert module(value)[0, 1, 0, 0] == 0.0


def test_residual_is_dense_finite_and_differentiable():
    module = LowRankClassResidual(64, 21, DLRBCConfig(mode="quadratic"))
    value = torch.randn(2, 64, 8, 8, requires_grad=True)
    output = module(value)
    assert output.shape == (2, 21, 8, 8)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert value.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_detect_wrapper_preserves_boxes_and_separates_dual_head_parameters():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        native = DLRBCDetectionModel(
            MODEL, nc=21, verbose=False, dlrbc=DLRBCConfig(mode="quadratic")
        )
    head = native.model[-1]
    assert isinstance(head, DLRBCDetectHead)
    assert head.class_tower_channels == (64, 64, 64)
    assert all(m.projection_channels == 32 for m in head.one2many_residuals)
    assert all(
        left.projection.weight.data_ptr() != right.projection.weight.data_ptr()
        for left, right in zip(head.one2many_residuals, head.one2one_residuals)
    )
    features = [
        torch.randn(1, 64, 8, 8),
        torch.randn(1, 128, 4, 4),
        torch.randn(1, 256, 2, 2),
    ]
    expected_boxes = [
        branch(feature)
        for branch, feature in zip(head.base_head.cv2, features)
    ]
    head.train()
    output = head([feature.clone() for feature in features])["one2many"]
    actual_boxes = torch.cat(
        [value.view(1, 4 * head.reg_max, -1) for value in expected_boxes], dim=-1
    )
    assert torch.equal(output["boxes"], actual_boxes)


def test_mode_changes_only_function_not_parameter_schema():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        linear = DLRBCDetectionModel(
            MODEL, nc=21, verbose=False, dlrbc=DLRBCConfig(mode="linear")
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        quadratic = DLRBCDetectionModel(
            MODEL, nc=21, verbose=False, dlrbc=DLRBCConfig(mode="quadratic")
        )
    assert list(linear.state_dict()) == list(quadratic.state_dict())
    assert all(
        torch.equal(linear.state_dict()[key], quadratic.state_dict()[key])
        for key in linear.state_dict()
    )
    linear.eval()
    quadratic.eval()
    probe = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        linear_output = linear(probe)[1]
        quadratic_output = quadratic(probe)[1]
    for branch in ("one2many", "one2one"):
        assert torch.equal(
            linear_output[branch]["boxes"], quadratic_output[branch]["boxes"]
        )
        assert not torch.equal(
            linear_output[branch]["scores"], quadratic_output[branch]["scores"]
        )


def test_dual_head_training_step_and_checkpoint_reload():
    model = DLRBCDetectionModel(
        MODEL, nc=21, verbose=False, dlrbc=DLRBCConfig(mode="quadratic")
    )
    model.train()
    output = model(torch.rand(2, 3, 64, 64))
    loss = sum(
        branch["scores"].square().mean() + branch["boxes"].square().mean()
        for branch in output.values()
    )
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.model[-1].one2many_residuals.parameters()
    )

    reloaded = DLRBCDetectionModel(
        MODEL, nc=21, verbose=False, dlrbc=DLRBCConfig(mode="quadratic")
    )
    reloaded.load_state_dict(model.state_dict(), strict=True)
    assert all(
        torch.equal(model.state_dict()[key], reloaded.state_dict()[key])
        for key in model.state_dict()
    )
