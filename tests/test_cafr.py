from pathlib import Path

import torch

from coffee_detector.cafr import (
    CAFRConfig,
    CAFRDetectionModel,
    CAFRInputEnhancer,
    calibrate_patch_size,
    choose_patch_size,
    frozen_variant_config,
    load_cafr_weights,
    shared_residual_gate,
    soft_spectral_weight,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def test_shared_gate_preserves_rgb_ratios_exactly():
    raw = torch.tensor([[[[0.2, 0.4]], [[0.1, 0.2]], [[0.05, 0.1]]]])
    recovered = torch.tensor([[[[2.0, 4.0]]]])
    out = shared_residual_gate(raw, recovered, eps=1e-8)
    assert torch.allclose(out[:, 0] / out[:, 1], raw[:, 0] / raw[:, 1], atol=1e-7, rtol=0.0)
    assert torch.allclose(out[:, 1] / out[:, 2], raw[:, 1] / raw[:, 2], atol=1e-7, rtol=0.0)


def test_soft_weight_is_monotonic_and_not_hard_zero():
    density = torch.tensor([[[[0.01, 0.25, 0.9]]]])
    threshold = torch.tensor([[[0.2]]])
    weight = soft_spectral_weight(density, threshold, temperature=0.05)
    assert 0.0 < weight[0, 0, 0, 0] < weight[0, 0, 0, 1] < weight[0, 0, 0, 2] < 1.0


def test_cafr_uses_three_equal_normalized_radial_bands():
    module = CAFRInputEnhancer(frozen_variant_config("CAFR", patch_size=32))
    radial = module.radial_bin
    assert set(radial.unique().tolist()) == {0, 1, 2}
    center = module.config.patch_size // 2
    assert radial[center, center].item() == 0
    assert radial[0, 0].item() == 2


def test_unsigned_orientation_maps_opposite_frequency_axes_together():
    module = CAFRInputEnhancer(frozen_variant_config("CAFR", patch_size=32))
    center = module.config.patch_size // 2
    # +x and -x are the same orientation for a real-image Fourier magnitude.
    assert module.angle_bin[center, center + 4].item() == module.angle_bin[center, center - 4].item()


def test_cafr_shape_dtype_finiteness_and_constant_identity():
    module = CAFRInputEnhancer(frozen_variant_config("CAFR", patch_size=32))
    image = torch.rand(1, 3, 65, 71, dtype=torch.float64)
    recovered = module.recover(image)
    output = module(image)
    assert recovered.shape == (1, 1, 65, 71)
    assert output.shape == image.shape
    assert recovered.dtype == image.dtype
    assert output.dtype == image.dtype
    assert torch.isfinite(recovered).all()
    assert torch.isfinite(output).all()

    constant = torch.full((1, 3, 64, 64), 0.4)
    assert torch.allclose(module(constant), constant, atol=1e-5, rtol=0.0)


def test_patch_calibration_frozen_rule(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    # At imgsz=640: equivalent sides are 25.6, 40.0, 64.0 pixels.
    (labels / "a.txt").write_text(
        "0 0.5 0.5 0.04 0.04\n"
        "1 0.5 0.5 0.0625 0.0625\n"
        "2 0.5 0.5 0.10 0.10\n",
        encoding="utf-8",
    )
    report = calibrate_patch_size(labels, imgsz=640, candidates=(16, 32, 64))
    assert report.boxes == 3
    assert abs(report.median_equivalent_side_px - 40.0) < 1e-9
    assert report.selected_patch_size == 32
    assert choose_patch_size(12.0, (16, 32, 64)) == 16
    assert choose_patch_size(80.0, (16, 32, 64)) == 64


def test_frozen_variant_ladder_changes_one_mechanism_at_a_time():
    c1 = frozen_variant_config("C1")
    c2 = frozen_variant_config("C2")
    c3 = frozen_variant_config("C3")
    c4 = frozen_variant_config("C4")
    assert c1.radial_bands == 1 and not c1.soft_selection
    assert c2.radial_bands == 3 and not c2.soft_selection
    assert c3.radial_bands == 3 and c3.soft_selection
    assert c1.angular_bins == c2.angular_bins == c3.angular_bins == 360
    assert c1.orientation_period == c2.orientation_period == c3.orientation_period == 360.0
    assert c4.angular_bins == 16 and c4.orientation_period == 180.0


def test_native_detector_state_is_bitwise_preserved_by_parameter_free_cafr():
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(11)
    source = DetectionModel(str(MODEL_YAML), nc=5, verbose=False).eval()
    candidate = CAFRDetectionModel(
        str(MODEL_YAML), nc=5, verbose=False, cafr=frozen_variant_config("CAFR", patch_size=32)
    ).eval()
    transfer = load_cafr_weights(candidate, source)
    source_state = source.state_dict()
    candidate_state = candidate.state_dict()
    assert source_state.keys() == candidate_state.keys()
    assert all(torch.equal(source_state[key], candidate_state[key]) for key in source_state)
    assert transfer["shape_compatible_items"] == len(source_state)


def test_cafr_is_active_on_training_and_inference_tensor_forwards():
    from torch import nn

    model = CAFRDetectionModel(
        str(MODEL_YAML), nc=5, verbose=False, cafr=frozen_variant_config("C1", patch_size=32)
    )

    class Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x

    spy = Spy()
    model.cafr = spy
    image = torch.randn(1, 3, 128, 128)
    model.train()
    train_output = model(image)
    assert isinstance(train_output, dict)
    assert spy.calls == 1
    model.eval()
    with torch.inference_mode():
        model(image)
    assert spy.calls == 2
