import numpy as np
import torch
from PIL import Image

from coffee_detector.experiments.run_faruq_v3_af2_illumination import (
    METRICS,
    summarize_illumination_rows,
)
from coffee_detector.illumination_stress import (
    CONDITIONS,
    CONDITION_BY_CODE,
    apply_illumination,
    apply_illumination_tensor,
)


def test_clean_is_pixel_exact_and_all_conditions_preserve_geometry():
    array = np.arange(17 * 23 * 3, dtype=np.uint8).reshape(17, 23, 3)
    image = Image.fromarray(array, mode="RGB")
    clean = np.asarray(apply_illumination(image, CONDITION_BY_CODE["clean"], key="x"))
    assert np.array_equal(clean, array)
    for condition in CONDITIONS:
        first = np.asarray(apply_illumination(image, condition, key="same"))
        second = np.asarray(apply_illumination(image, condition, key="same"))
        assert first.shape == array.shape
        assert np.array_equal(first, second)


def test_exposure_directions_change_mean_as_intended():
    image = Image.fromarray(np.full((20, 30, 3), 100, dtype=np.uint8), mode="RGB")
    clean = np.asarray(image).mean()
    dark = np.asarray(
        apply_illumination(image, CONDITION_BY_CODE["dark_ev10"], key="x")
    ).mean()
    bright = np.asarray(
        apply_illumination(image, CONDITION_BY_CODE["bright_ev10"], key="x")
    ).mean()
    assert dark < clean < bright


def test_tensor_path_is_exact_for_clean_and_geometry_preserving_for_stress():
    images = torch.rand(3, 3, 31, 47)
    clean = apply_illumination_tensor(
        images, CONDITION_BY_CODE["clean"], keys=["a", "b", "c"]
    )
    assert torch.equal(clean, images)
    for condition in CONDITIONS:
        first = apply_illumination_tensor(images, condition, keys=["a", "b", "c"])
        second = apply_illumination_tensor(images, condition, keys=["a", "b", "c"])
        assert first.shape == images.shape
        assert torch.equal(first, second)
        assert torch.isfinite(first).all()
        assert first.min() >= 0 and first.max() <= 1


def test_summary_uses_clean_normalized_degradation_and_passes_consistent_gain():
    rows = []
    for condition in CONDITIONS:
        for model, clean in (("D0FT", 0.80), ("AF2", 0.82)):
            degradation = 0.0 if condition.is_clean else (-0.10 if model == "D0FT" else -0.05)
            metrics = {metric: clean + degradation for metric in METRICS}
            rows.append(
                {
                    "seed": 42,
                    "condition": condition.code,
                    "model": model,
                    "metrics": metrics,
                }
            )
    result = summarize_illumination_rows(rows, [42])
    assert result["decision"] == "PASS"
    assert all(result["criteria"].values())
    macro = next(row for row in result["aggregate"] if row["metric"] == "macro_map50_95")
    assert abs(macro["mean_robustness_advantage"] - 0.05) < 1e-9
    assert macro["positive_pairs"] == 9
