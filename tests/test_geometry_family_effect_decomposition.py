import json
from pathlib import Path

from coffee_detector.analysis.geometry_family_effect_decomposition import (
    FAMILIES,
    PROTOCOL,
    SOURCE_PROTOCOL,
    decompose_geometry_family_effects,
)


def _source(tmp_path: Path) -> Path:
    per_seed = {}
    seed_values = {
        42: {
            "kulit_kopi": (-0.01, -0.02, -0.03),
            "kulit_tanduk": (0.04, 0.03, 0.02),
            "tanah_batu_ranting": (-0.01, 0.01, 0.00),
        },
        123: {
            "kulit_kopi": (-0.02, -0.01, -0.02),
            "kulit_tanduk": (0.02, 0.02, 0.01),
            "tanah_batu_ranting": (-0.02, 0.02, 0.00),
        },
        2026: {
            "kulit_kopi": (-0.01, -0.01, -0.01),
            "kulit_tanduk": (0.01, 0.02, 0.01),
            "tanah_batu_ranting": (0.01, 0.03, 0.02),
        },
    }
    for seed, families in seed_values.items():
        control = {}
        geometry = {}
        for family, classes in FAMILIES.items():
            for name, delta in zip(classes, families[family]):
                control[name] = 0.80
                geometry[name] = 0.80 + delta
        per_seed[str(seed)] = {
            "results": {
                "GEO-C0": {"size_map50_95_by_class": control},
                "GEO1": {"size_map50_95_by_class": geometry},
            }
        }
    payload = {
        "protocol": SOURCE_PROTOCOL,
        "seeds": [42, 123, 2026],
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "decision": "PASS",
        "per_seed": per_seed,
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_family_decomposition_preserves_posthoc_guardrail(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "out.json"
    result = decompose_geometry_family_effects(source, output)
    assert result["protocol"] == PROTOCOL
    assert result["analysis_status"] == "posthoc_descriptive_decomposition_no_new_gate"
    assert result["test_opened"] is False
    assert output.is_file()


def test_family_patterns_detect_consistent_positive_and_negative(tmp_path):
    result = decompose_geometry_family_effects(_source(tmp_path), tmp_path / "out.json")
    assert result["aggregate_families"]["kulit_kopi"]["pattern"] == "negative_3_of_3"
    assert result["aggregate_families"]["kulit_tanduk"]["pattern"] == "positive_3_of_3"
    assert result["aggregate_families"]["tanah_batu_ranting"]["pattern"] == "mixed_across_seeds"
    contrast = result["family_contrasts"]["kulit_tanduk_minus_kulit_kopi"]
    assert contrast["positive_seeds"] == 3


def test_family_means_are_means_of_three_classes(tmp_path):
    result = decompose_geometry_family_effects(_source(tmp_path), tmp_path / "out.json")
    seed42 = result["per_seed"]["42"]["families"]
    assert abs(seed42["kulit_kopi"]["mean_delta"] - (-0.02)) < 1e-12
    assert abs(seed42["kulit_tanduk"]["mean_delta"] - 0.03) < 1e-12
