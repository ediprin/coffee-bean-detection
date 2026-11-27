from types import SimpleNamespace

import pytest
import torch
from ultralytics.models.yolo.detect import DetectionValidator

from coffee_detector.auliya_mask2 import CommonMaskGridValidator, prototypes_at_grid


def test_stride2_identity_roundtrips_to_old_grid():
    coarse = torch.randn((1, 32, 24, 32), generator=torch.Generator().manual_seed(4))
    fine = coarse.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    assert torch.equal(prototypes_at_grid(fine, (96, 128), 4), coarse)
    assert prototypes_at_grid(fine, (96, 128), 2) is fine
    assert prototypes_at_grid(coarse, (96, 128), 4) is coarse
    assert prototypes_at_grid(coarse, (96, 128), 2).shape == fine.shape
    with pytest.raises(ValueError, match="Unexpected prototype"):
        prototypes_at_grid(coarse, (128, 128), 4)


@pytest.mark.parametrize("ratio", [2, 4])
def test_common_grid_uses_image_geometry_for_box_cropping(monkeypatch, ratio):
    def detections(_, _raw):
        return [{"extra": torch.ones(1, 1), "bboxes": torch.tensor([[8., 16., 24., 32.]])}]

    monkeypatch.setattr(DetectionValidator, "postprocess", detections)
    validator = object.__new__(CommonMaskGridValidator)
    validator.input_hw = (64, 96)
    validator.args = SimpleNamespace(mask_ratio=ratio, tile=False, save_json=False, save_txt=False)
    # Candidate always emits stride 2, including when the metric grid is 4.
    masks = validator.postprocess(((torch.empty(0), torch.ones(1, 1, 32, 48)), {}))[0]["masks"]
    assert masks.shape == (1, 64 // ratio, 96 // ratio)
    y, x = torch.where(masks[0])
    assert (int(x.min()), int(y.min()), int(x.max()), int(y.max())) == (
        8 // ratio, 16 // ratio, 24 // ratio - 1, 32 // ratio - 1,
    )
