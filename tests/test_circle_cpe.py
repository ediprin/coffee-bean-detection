import json
from pathlib import Path

import torch

from coffee_detector.circle_cpe import CircleCPEConfig, circle_pair_loss


ROOT = Path(__file__).resolve().parents[1]


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


def test_colab_v2_reuses_control_reports_without_requiring_old_checkpoints():
    notebook = ROOT / "notebooks/Faruq_V3_Circle_CPE_Matched_Objective_Screening_Colab_v2.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    report = "CPE0_REPORT=CONTROL_REPORT_DIR/'CPE0_seed42_val.json'"
    conditional = (
        "CPE0_CHECKPOINT=None if CPE0_REPORT.is_file() "
        "else find_unique_checkpoint('CPE0_seed42')"
    )
    assert report in source
    assert conditional in source
    assert source.index(report) < source.index(conditional)
