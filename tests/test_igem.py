import json
from pathlib import Path

import torch
import yaml

from coffee_detector.igem import (
    IGEMConfig,
    IGEMDetectionModel,
    IGEMDetectHead,
    load_igem_detector_weights,
    rectangular_class_mask_targets,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(nc: int = 5):
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(41)
    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = IGEMDetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        igem=IGEMConfig(),
    ).eval()
    load_igem_detector_weights(candidate, source)
    return source, candidate


def test_igem_identity_start_preserves_native_inference_exactly():
    source, candidate = _models()
    assert isinstance(candidate.model[-1], IGEMDetectHead)
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        native = source(image)
        transferred = candidate(image)
    assert torch.equal(transferred[0], native[0])
    assert torch.equal(transferred[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"])
    assert torch.equal(transferred[1]["one2one"]["scores"], native[1]["one2one"]["scores"])


def test_igem_training_exposes_masks_only_on_one2many_and_keeps_native_boxes():
    source, candidate = _models()
    source.train(); candidate.train()
    image = torch.randn(2, 3, 128, 128)
    native = source(image)
    transferred = candidate(image)
    assert "igem_mask_logits" in transferred["one2many"]
    assert "igem_mask_logits" not in transferred["one2one"]
    assert len(transferred["one2many"]["igem_mask_logits"]) == 3
    for branch in ("one2many", "one2one"):
        assert torch.equal(transferred[branch]["boxes"], native[branch]["boxes"])
        assert torch.equal(transferred[branch]["scores"], native[branch]["scores"])


def test_rectangular_mask_targets_use_background_n_and_foreground_class():
    logits = torch.zeros(1, 4, 8, 8)  # three classes + background id=3
    batch = {
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "cls": torch.tensor([[1.0]]),
        "batch_idx": torch.tensor([0.0]),
    }
    target = rectangular_class_mask_targets(logits, batch, num_classes=3)
    assert target.shape == (1, 8, 8)
    assert target[0, 3:5, 3:5].eq(1).all()
    assert target[0, 0, 0].item() == 3


def test_config_freezes_paper_values_and_declared_transfer_choices():
    payload = yaml.safe_load((ROOT / "configs/igem/IGEM1_yolo26n_classification_guidance.yaml").read_text())
    config = payload["igem"]
    assert config["reference_depth"] == 3
    assert config["mask_loss_weight"] == 0.05
    assert config["kernel_size"] == 3
    assert config["attention_heads"] == 4
    assert config["channel_reduction"] == 4
    assert payload["train"]["epochs"] == 50
    assert payload["train"]["imgsz"] == 640


def test_notebook_is_branch_correct_and_val_only():
    path = ROOT / "notebooks/Faruq_V3_IGEM_Screening_Colab.ipynb"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
    assert "agent/igem-classification-guidance-screening" in source
    assert "run_faruq_v3_igem_screening" in source
    assert "--authorize-training" in source
    assert "split=test" not in source.lower()
    assert "--split test" not in source.lower()
