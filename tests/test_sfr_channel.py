import json
from pathlib import Path

import torch

from coffee_detector.sfr_channel import (
    LSHChannelFormer,
    SFRChannelConfig,
    SFRChannelDetectHead,
    SFRChannelDetectionModel,
    load_sfr_channel_weights,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(mode="channel", nc=5):
    from ultralytics.nn.tasks import DetectionModel
    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    config = SFRChannelConfig(
        hidden_dim=32, spatial_heads=4, window_size=7, bucket_count=4,
        hash_seed=2023, mode=mode,
    )
    candidate = SFRChannelDetectionModel(
        str(MODEL_YAML), nc=nc, verbose=False, sfr_channel=config
    ).eval()
    load_sfr_channel_weights(candidate, source)
    return source, candidate


def test_lsh_channel_former_restores_shape_and_is_finite():
    block = LSHChannelFormer(16, SFRChannelConfig(hidden_dim=32, bucket_count=4))
    value = torch.randn(2, 16, 13, 17)
    output = block(value)
    assert output.shape == (2, 32, 13, 17)
    assert torch.isfinite(output).all()


def test_lsh_hash_is_reproducible_and_qk_is_shared():
    config = SFRChannelConfig(hidden_dim=32, bucket_count=4, hash_seed=77)
    first = LSHChannelFormer(8, config)
    second = LSHChannelFormer(8, config)
    assert torch.equal(first.hash_projection, second.hash_projection)
    assert hasattr(first, "shared_qk")
    assert not hasattr(first, "query")
    assert not hasattr(first, "key")
    tokens = torch.randn(3, 32, 49)
    assert torch.equal(first.bucket_order(tokens), first.bucket_order(tokens))


def test_bucket_order_partitions_every_channel_once():
    block = LSHChannelFormer(8, SFRChannelConfig(hidden_dim=32, bucket_count=4))
    tokens = torch.randn(2, 32, 49)
    order = block.bucket_order(tokens)
    expected = torch.arange(32).expand(2, -1)
    assert torch.equal(torch.sort(order, dim=1).values, expected)
    assert order.shape[1] // block.config.bucket_count == 8


def test_c1_starts_at_native_d0_and_preserves_boxes():
    source, candidate = _models("channel")
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        source_output = source(image)
        candidate_output = candidate(image)
    assert isinstance(candidate.model[-1], SFRChannelDetectHead)
    assert torch.allclose(candidate_output[0], source_output[0], rtol=0.0, atol=1e-7)
    assert torch.equal(candidate_output[1]["one2one"]["boxes"], source_output[1]["one2one"]["boxes"])
    assert torch.equal(candidate_output[1]["one2one"]["scores"], source_output[1]["one2one"]["scores"])


def test_active_c1_changes_scores_without_changing_boxes():
    _, candidate = _models("channel")
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        before = candidate(image)
    head = candidate.model[-1]
    with torch.no_grad():
        head.correction.classifiers[0].bias.fill_(0.2)
    with torch.inference_mode():
        after = candidate(image)
    assert torch.equal(before[1]["one2one"]["boxes"], after[1]["one2one"]["boxes"])
    assert not torch.equal(before[1]["one2one"]["scores"], after[1]["one2one"]["scores"])


def test_sc1_contains_parallel_spatial_and_channel_blocks():
    _, candidate = _models("spatial_channel")
    correction = candidate.model[-1].correction
    assert correction.spatial_blocks is not None
    assert len(correction.spatial_blocks) == 3
    assert len(correction.channel_blocks) == 3


def test_training_gradient_reaches_channel_qk_projection():
    _, candidate = _models("channel")
    candidate.train()
    output = candidate(torch.randn(1, 3, 128, 128))
    loss = output["one2many"]["scores"].square().mean()
    loss.backward()
    parameter = candidate.model[-1].correction.channel_blocks[0].shared_qk.weight
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


def test_configs_are_frozen_to_paper_bucket_count_and_modes():
    import yaml
    c1 = yaml.safe_load((ROOT / "configs/sfr_channel/C1_yolo26n_lsh_channel_former.yaml").read_text())
    sc1 = yaml.safe_load((ROOT / "configs/sfr_channel/SC1_yolo26n_spatial_channel_former.yaml").read_text())
    assert c1["sfr_channel"]["bucket_count"] == 4
    assert sc1["sfr_channel"]["bucket_count"] == 4
    assert c1["sfr_channel"]["mode"] == "channel"
    assert sc1["sfr_channel"]["mode"] == "spatial_channel"


def test_notebook_is_val_only_and_points_to_branch():
    notebook = ROOT / "notebooks/Faruq_V3_SFR_Channel_SC_Screening_Colab.ipynb"
    assert notebook.is_file()
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
    assert "agent/sfrnet-channel-former-screening" in source
    assert "run_faruq_v3_sfr_channel_screening" in source
    assert "--authorize-training" in source
    assert "split=test" not in source.lower()
