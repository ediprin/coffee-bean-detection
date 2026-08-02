from pathlib import Path

from coffee_detector.multilevel_head.audit import static_multilevel_head_audit


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def test_static_audit_preserves_capacity_and_contracts(tmp_path: Path) -> None:
    result = static_multilevel_head_audit(
        MODEL_YAML,
        tmp_path / "audit.json",
        nc=5,
        weights=None,
        image_size=64,
        topk=4,
    )
    assert result["training_executed"] is False
    assert result["dataset_accessed"] is False
    assert result["test_images_accessed"] is False
    assert result["gates"]["same_parameter_count"] is True
    assert result["gates"]["same_state_dict_schema"] is True
    assert result["gates"]["native_heads_preserved"] is True
    assert result["gates"]["control_zero_is_d0"] is True
    assert result["gates"]["fusion_zero_is_d0"] is True
    assert result["identity"]["absolute_tolerance"] == 1e-7
    assert result["identity"]["control_zero_vs_d0_max_abs_diff"] <= 1e-7
    assert result["identity"]["fusion_zero_vs_d0_max_abs_diff"] <= 1e-7
    assert result["gates"]["zero_raw_native_predictions_bitwise_equal"] is True
    assert result["gates"]["active_modes_differ"] is True
    assert result["finite_refiner_gradients"] is True
    assert result["state_roundtrip_equal"] is True
