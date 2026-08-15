import json
from pathlib import Path

import torch

from coffee_detector.experiments.run_faruq_v3_geometry_conditioning_screen import (
    _control_validity,
    _geometry_gate,
)
from coffee_detector.geometry_conditioning.model import (
    GeometryConditionedDetectionModel,
    GeometryConditioningConfig,
    GeometryLogitAdapter,
    load_geometry_conditioned_weights,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _names():
    values = {index: f"class_{index}" for index in range(21)}
    values[1] = "kulit_kopi_ukuran_kecil"
    values[2] = "kulit_kopi_ukuran_sedang"
    values[3] = "kulit_kopi_ukuran_besar"
    values[4] = "kulit_tanduk_ukuran_kecil"
    values[5] = "kulit_tanduk_ukuran_sedang"
    return values


def test_adapter_is_zero_start_and_masks_non_size_classes():
    adapter = GeometryLogitAdapter(21, GeometryConditioningConfig(), _names())
    geometry = torch.rand(2, 4, 19)
    with torch.inference_mode():
        output = adapter(geometry)
    assert torch.count_nonzero(output) == 0
    mask = adapter.class_mask.flatten()
    assert mask[0] == 0
    assert mask[1] == 1 and mask[5] == 1
    assert int(mask.sum().item()) == 5


def test_geometry_and_zero_control_have_exact_same_parameter_count_and_native_start():
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=21, verbose=False).eval()
    control = GeometryConditionedDetectionModel(
        str(MODEL_YAML),
        nc=21,
        verbose=False,
        geometry_conditioning=GeometryConditioningConfig(),
        signal_mode="zero",
        class_names=_names(),
    ).eval()
    geometry = GeometryConditionedDetectionModel(
        str(MODEL_YAML),
        nc=21,
        verbose=False,
        geometry_conditioning=GeometryConditioningConfig(),
        signal_mode="geometry",
        class_names=_names(),
    ).eval()
    load_geometry_conditioned_weights(control, source)
    load_geometry_conditioned_weights(geometry, source)
    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in geometry.parameters()
    )
    image = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        native, c0, geo = source(image), control(image), geometry(image)
    for candidate in (c0, geo):
        assert torch.equal(native[0], candidate[0])
        assert torch.equal(
            native[1]["one2one"]["boxes"], candidate[1]["one2one"]["boxes"]
        )
        assert torch.equal(
            native[1]["one2one"]["scores"], candidate[1]["one2one"]["scores"]
        )


def test_frozen_screening_gate_requires_geometry_specific_and_size_gain():
    d0ft = {
        "macro_map50_95": 0.866,
        "bottom3_class_map50_95": 0.750,
        "worst_class_map50_95": 0.720,
    }
    control = {
        "macro_map50_95": 0.868,
        "bottom3_class_map50_95": 0.770,
        "worst_class_map50_95": 0.750,
        "size_class_mean_map50_95": 0.800,
    }
    geo = {
        "macro_map50_95": 0.872,
        "bottom3_class_map50_95": 0.777,
        "worst_class_map50_95": 0.752,
        "size_class_mean_map50_95": 0.807,
    }
    assert _control_validity(control, d0ft)["decision"] == "PASS"
    assert _geometry_gate(geo, control, d0ft)["decision"] == "PASS"
    failed = dict(geo, size_class_mean_map50_95=0.803)
    assert _geometry_gate(failed, control, d0ft)["decision"] == "FAIL"


def test_colab_uses_compact_progress_instead_of_streaming_ultralytics_output():
    notebook = ROOT / "notebooks/Faruq_V3_Geometry_Conditioning_Screening_Colab.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "TRAIN_LOG = Path('/content/geometry_conditioning_train.log')" in source
    assert "def compact_progress():" in source
    assert "time.sleep(60)" in source
    assert "for line in process.stdout" not in source
