import torch

from coffee_detector.circle_cpe import CircleCPEConfig, circle_pair_loss


def test_circle_loss_prefers_separated_class_geometry():
    labels = torch.tensor([0, 0, 1, 1])
    good = torch.tensor([
        [1.0, 0.0], [0.98, 0.02], [-1.0, 0.0], [-0.98, -0.02]
    ])
    bad = torch.tensor([
        [1.0, 0.0], [-1.0, 0.0], [0.98, 0.02], [-0.98, -0.02]
    ])
    good_loss = circle_pair_loss(good, labels, margin=0.25, gamma=32.0)
    bad_loss = circle_pair_loss(bad, labels, margin=0.25, gamma=32.0)
    assert torch.isfinite(good_loss)
    assert torch.isfinite(bad_loss)
    assert good_loss < bad_loss


def test_circle_loss_zero_without_both_pair_types():
    embeddings = torch.randn(3, 8)
    labels = torch.zeros(3, dtype=torch.long)
    loss = circle_pair_loss(embeddings, labels)
    assert loss.item() == 0.0


def test_circle_config_projection_is_inference_matched_to_cpe():
    config = CircleCPEConfig(
        embedding_dim=128, iou_threshold=0.7, margin=0.25, gamma=256.0, loss_weight=0.005
    )
    projection = config.projection_config()
    assert projection.embedding_dim == 128
    assert projection.iou_threshold == 0.7
    assert projection.temperature == 0.2
    assert projection.loss_weight == 0.005
