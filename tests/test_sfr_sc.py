import json
from pathlib import Path

import torch

from coffee_detector.sfr_sc import (
    SFRSCConfig,
    SFRSCDetectionModel,
    SFRSCDetectHead,
    WindowChannelLSHFormer,
    load_sfr_sc_weights,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(nc=5):
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = SFRSCDetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        sfr_sc=SFRSCConfig(hidden_dim=32, num_heads=4, window_size=7, hash_buckets=4),
    ).eval()
    load_sfr_sc_weights(candidate, source)
    return source, candidate


def test_channel_former_restores_nondivisible_shape():
    block = WindowChannelLSHFormer(
        16, SFRSCConfig(hidden_dim=32, num_heads=4, window_size=7, hash_buckets=4)
    )
    value = torch.randn(2, 16, 13, 17)
    output = block(value)
    assert output.shape == (2, 32, 13, 17)
    assert torch.isfinite(output).all()


def test_channel_former_uses_one_shared_qk_projection_and_four_buckets():
    config = SFRSCConfig(hidden_dim=32, hash_buckets=4, hash_seed=2023)
    block = WindowChannelLSHFormer(16, config)
    assert hasattr(block, "qk")
    assert not hasattr(block, "query")
    assert not hasattr(block, "key")
    tokens = torch.randn(3, 32, 49)
    first = block.hash_tokens(tokens)
    second = block.hash_tokens(tokens)
    assert torch.equal(first, second)
    assert first.min().item() >= 0
    assert first.max().item() < 4
    assert block.hash_projection.shape == (49, 2)


def test_hash_projection_is_fixed_not_trainable():
    block = WindowChannelLSHFormer(8, SFRSCConfig(hidden_dim=16, hash_buckets=4))
    named_parameters = dict(block.named_parameters())
    named_buffers = dict(block.named_buffers())
    assert "hash_projection" not in named_parameters
    assert "hash_projection" in named_buffers


def test_sc1_starts_at_native_d0_and_preserves_boxes_and_scores():
    source, candidate = _models()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        source_output = source(image)
        candidate_output = candidate(image)
    assert isinstance(candidate.model[-1], SFRSCDetectHead)
    assert torch.allclose(candidate_output[0], source_output[0], rtol=0.0, atol=1e-7)
    assert torch.equal(
        candidate_output[1]["one2one"]["boxes"], source_output[1]["one2one"]["boxes"]
    )
    assert torch.equal(
        candidate_output[1]["one2one"]["scores"], source_output[1]["one2one"]["scores"]
    )


def test_active_sc1_changes_scores_without_changing_boxes():
    _, candidate = _models()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        before = candidate(image)
    head = candidate.model[-1]
    with torch.no_grad():
        head.sc.classifiers[0].bias.fill_(0.2)
    with torch.inference_mode():
        after = candidate(image)
    assert torch.equal(
        before[1]["one2one"]["boxes"], after[1]["one2one"]["boxes"]
    )
    assert not torch.equal(
        before[1]["one2one"]["scores"], after[1]["one2one"]["scores"]
    )


def test_training_gradient_reaches_spatial_and_channel_paths():
    _, candidate = _models()
    candidate.train()
    output = candidate(torch.randn(1, 3, 128, 128))
    loss = output["one2many"]["scores"].square().mean()
    loss.backward()
    head = candidate.model[-1]
    spatial = head.sc.spatial_blocks[0]
    channel = head.sc.channel_blocks[0]
    assert spatial.attn.in_proj_weight.grad is not None
    assert channel.qk.weight.grad is not None
    assert torch.isfinite(spatial.attn.in_proj_weight.grad).all()
    assert torch.isfinite(channel.qk.weight.grad).all()


def test_config_freezes_paper_default_bucket_count():
    config = SFRSCConfig.from_mapping({})
    assert config.hash_buckets == 4
    assert config.window_size == 7


def test_sc_notebook_is_val_only_and_points_to_branch():
    notebook = ROOT / "notebooks/Faruq_V3_SFR_SC_Former_Screening_Colab.ipynb"
    assert notebook.is_file()
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "agent/sfrnet-sc-former-screening" in source
    assert "run_faruq_v3_sfr_sc_screening" in source
    assert "--authorize-training" in source
    assert "split=test" not in source.lower()
    assert "test/" not in source.lower()
