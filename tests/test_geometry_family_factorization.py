from pathlib import Path

import torch

from coffee_detector.experiments.run_faruq_v3_geometry_family_factorization import _aggregate
from coffee_detector.geometry_factorization.model import (
    FAMILIES,
    Family35x3GeometryAdapter,
    GeometryFactorizationConfig,
    GeometryFactorizedDetectionModel,
    Shared60GeometryAdapter,
    family_class_indices,
    load_geometry_factorized_weights,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _names():
    values = [f"dummy_{i}" for i in range(21)]
    targets = [
        "kulit_kopi_ukuran_kecil",
        "kulit_kopi_ukuran_sedang",
        "kulit_kopi_ukuran_besar",
        "kulit_tanduk_ukuran_kecil",
        "kulit_tanduk_ukuran_sedang",
        "kulit_tanduk_ukuran_besar",
        "tanah_batu_ranting_kecil",
        "tanah_batu_ranting_sedang",
        "tanah_batu_ranting_besar",
    ]
    for index, name in enumerate(targets):
        values[index] = name
    return {index: name for index, name in enumerate(values)}


def test_family_mapping_is_exact_three_by_three():
    mapping = family_class_indices(_names(), 21)
    assert tuple(mapping) == FAMILIES
    assert mapping["kulit_kopi"] == (0, 1, 2)
    assert mapping["kulit_tanduk"] == (3, 4, 5)
    assert mapping["tanah_batu_ranting"] == (6, 7, 8)


def test_shared_and_family_adapters_have_exact_same_849_parameters():
    config = GeometryFactorizationConfig()
    shared = Shared60GeometryAdapter(21, config, _names())
    family = Family35x3GeometryAdapter(21, config, _names())
    shared_params = sum(parameter.numel() for parameter in shared.parameters())
    family_params = sum(parameter.numel() for parameter in family.parameters())
    assert shared_params == family_params == 849
    geometry = torch.rand(2, 4, 17)
    with torch.no_grad():
        left, right = shared(geometry), family(geometry)
    assert left.shape == right.shape == (2, 21, 17)
    assert torch.count_nonzero(left) == 0
    assert torch.count_nonzero(right) == 0


def test_factorized_models_begin_exactly_from_native_function():
    from ultralytics.nn.tasks import DetectionModel

    names = _names()
    source = DetectionModel(str(MODEL_YAML), nc=21, verbose=False).eval()
    shared = GeometryFactorizedDetectionModel(
        str(MODEL_YAML), nc=21, verbose=False,
        geometry_factorization=GeometryFactorizationConfig(), mode="shared60", class_names=names,
    ).eval()
    family = GeometryFactorizedDetectionModel(
        str(MODEL_YAML), nc=21, verbose=False,
        geometry_factorization=GeometryFactorizationConfig(), mode="family35x3", class_names=names,
    ).eval()
    load_geometry_factorized_weights(shared, source)
    load_geometry_factorized_weights(family, source)
    image = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        native, shared_zero, family_zero = source(image), shared(image), family(image)
    for candidate in (shared_zero, family_zero):
        assert torch.equal(native[1]["one2one"]["boxes"], candidate[1]["one2one"]["boxes"])
        assert torch.equal(native[1]["one2one"]["scores"], candidate[1]["one2one"]["scores"])


def _record(macro, bottom3, worst, size, coffee, horn, tbr):
    return {
        "fam_minus_shared": {
            "macro_map50_95": macro,
            "bottom3_class_map50_95": bottom3,
            "worst_class_map50_95": worst,
            "size_class_mean_map50_95": size,
        },
        "family_deltas": {
            "kulit_kopi": coffee,
            "kulit_tanduk": horn,
            "tanah_batu_ranting": tbr,
        },
    }


def test_exploratory_gate_passes_reproducible_family_factorization():
    per_seed = {
        "42": _record(.003, .010, .012, .008, .010, .001, .004),
        "123": _record(.004, .006, .004, .007, .007, -.002, .001),
        "2026": _record(.002, .002, .003, .006, .006, .001, -.001),
    }
    _, _, criteria = _aggregate(per_seed)
    assert all(criteria.values())


def test_exploratory_gate_rejects_no_coffee_recovery():
    per_seed = {
        "42": _record(.003, .010, .012, .008, .001, .001, .004),
        "123": _record(.004, .006, .004, .007, .000, -.002, .001),
        "2026": _record(.002, .002, .003, .006, -.001, .001, -.001),
    }
    _, _, criteria = _aggregate(per_seed)
    assert criteria["kulit_kopi_family_mean_gain_at_least_0_5_point"] is False
    assert not all(criteria.values())
