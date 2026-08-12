import json
from pathlib import Path

import torch
from torch import nn

from coffee_detector.afab import (
    AFABConfig,
    AFABDetectionModel,
    AFABInputEnhancer,
    af2_entropy_threshold,
    afab_gate,
    load_afab_weights,
    minmax_spatial,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def test_af2_entropy_threshold_matches_eq10_eq11_for_uniform_density():
    probability = torch.full((1, 1, 4), 0.25)
    threshold = af2_entropy_threshold(probability, gamma=0.1)
    # H=ln(4), exp(-H)=1/4, so t=0.1/(1+1/4)=0.08.
    assert torch.allclose(threshold, torch.tensor([[0.08]]), atol=1e-7, rtol=0.0)


def test_af1_adaptive_highpass_suppresses_dc_and_keeps_radius_one_for_equal_energy():
    module = AFABInputEnhancer(AFABConfig(mode="af1", patch_size=32, overlap=0.5))
    energy = torch.ones(1, 1)
    mask = module._af1_mask(energy, energy)
    center = module.config.patch_size // 2
    assert mask[0, 0, center, center].item() == 0.0
    assert mask[0, 0, center, center + 1].item() == 1.0


def test_af2_suppresses_low_density_direction():
    module = AFABInputEnhancer(AFABConfig(mode="af2", angular_bins=360))
    m = module.config.patch_size
    center = m // 2
    frequency = torch.zeros(1, 1, m, m, dtype=torch.complex64)
    frequency[0, 0, center, center + 4] = 10.0 + 0.0j  # 0 degrees
    frequency[0, 0, center + 4, center] = 0.001 + 0.0j  # 90 degrees
    weight = module._af2_weight(frequency)
    assert weight[0, 0, center, center + 4] > 0.99
    assert weight[0, 0, center + 4, center] == 0.0


def test_minmax_and_gate_follow_described_fusion_exactly():
    raw = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]])
    recovered = torch.tensor([[[[2.0, 4.0], [6.0, 10.0]]]])
    normalized = minmax_spatial(recovered)
    expected_norm = torch.tensor([[[[0.0, 0.25], [0.5, 1.0]]]])
    assert torch.allclose(normalized, expected_norm, atol=1e-7, rtol=0.0)
    assert torch.allclose(afab_gate(raw, recovered), raw + raw * expected_norm, atol=1e-7, rtol=0.0)


def test_overlap_reconstruction_shape_dtype_and_finiteness():
    module = AFABInputEnhancer(
        AFABConfig(mode="af12", patch_size=32, overlap=0.5, chunk_size=8)
    )
    image = torch.rand(1, 3, 65, 71, dtype=torch.float64)
    recovered = module.recover(image)
    output = module(image)
    assert recovered.shape == image.shape
    assert output.shape == image.shape
    assert recovered.dtype == image.dtype
    assert output.dtype == image.dtype
    assert torch.isfinite(recovered).all()
    assert torch.isfinite(output).all()


def test_constant_image_af1_gate_is_identity_after_dc_suppression():
    module = AFABInputEnhancer(AFABConfig(mode="af1", patch_size=32, overlap=0.5))
    image = torch.full((1, 3, 64, 64), 0.4)
    output = module(image)
    assert torch.allclose(output, image, atol=1e-5, rtol=0.0)


def test_native_detector_weights_transfer_bitwise_and_afab_adds_no_persistent_state():
    from ultralytics.nn.tasks import DetectionModel
    torch.manual_seed(11)
    source = DetectionModel(str(MODEL_YAML), nc=5, verbose=False).eval()
    candidate = AFABDetectionModel(
        str(MODEL_YAML),
        nc=5,
        verbose=False,
        afab=AFABConfig(mode="af1", patch_size=32, overlap=0.5),
    ).eval()
    transfer = load_afab_weights(candidate, source)
    source_state = source.state_dict()
    candidate_state = candidate.state_dict()
    assert source_state.keys() == candidate_state.keys()
    assert all(torch.equal(source_state[key], candidate_state[key]) for key in source_state)
    assert transfer["shape_compatible_items"] == len(source_state)


def test_afab_predict_path_is_active_in_train_and_eval_tensor_forwards():
    model = AFABDetectionModel(
        str(MODEL_YAML),
        nc=5,
        verbose=False,
        afab=AFABConfig(mode="af1", patch_size=32, overlap=0.5),
    )

    class Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, x):
            self.calls += 1
            return x

    spy = Spy()
    model.afab = spy
    image = torch.randn(1, 3, 128, 128)
    model.train()
    train_output = model(image)
    assert isinstance(train_output, dict)
    assert spy.calls == 1
    model.eval()
    with torch.inference_mode():
        model(image)
    assert spy.calls == 2


def test_breadth_configs_freeze_paper_and_transfer_settings():
    import yaml
    for arm, mode in (("AF1", "af1"), ("AF2", "af2"), ("AF12", "af12")):
        path = next((ROOT / "configs/afab").glob(f"{arm}_*.yaml"))
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = payload["afab"]
        assert cfg["mode"] == mode
        assert cfg["patch_size"] == 32
        assert cfg["overlap"] == 0.50
        assert cfg["radius_ratio"] == 0.05
        assert cfg["gamma"] == 0.10
        assert cfg["angular_bins"] == 360

# The synthesis branch reuses the audited AFAB operator; its executable
# contract is covered by the AGSF notebook rather than the standalone AFAB one.
