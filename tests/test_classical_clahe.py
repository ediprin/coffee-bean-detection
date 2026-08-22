from __future__ import annotations

import pytest
import torch

from coffee_detector.classical_enhancement import CLAHEConfig, CLAHEInputEnhancer


def test_clahe_config_is_frozen_and_serializable():
    config = CLAHEConfig.from_mapping({"clip_limit": 3.0, "tile_grid_size": [8, 8]})
    assert config.clip_limit == 3.0
    assert config.tile_grid_size == (8, 8)
    assert config.to_dict() == {"clip_limit": 3.0, "tile_grid_size": [8, 8]}


def test_clahe_rejects_invalid_config():
    with pytest.raises(ValueError):
        CLAHEConfig.from_mapping({"clip_limit": 0.0, "tile_grid_size": [8, 8]})
    with pytest.raises(ValueError):
        CLAHEConfig.from_mapping({"clip_limit": 3.0, "tile_grid_size": [8, 0]})


def test_clahe_forward_preserves_shape_dtype_and_range():
    pytest.importorskip("cv2")
    torch.manual_seed(7)
    image = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    enhancer = CLAHEInputEnhancer(CLAHEConfig())
    output = enhancer(image)
    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert output.device == image.device
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


def test_clahe_forward_is_deterministic():
    pytest.importorskip("cv2")
    torch.manual_seed(19)
    image = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    enhancer = CLAHEInputEnhancer(CLAHEConfig())
    first = enhancer(image)
    second = enhancer(image)
    assert torch.equal(first, second)
