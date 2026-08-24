from coffee_detector.analysis.ontology_error_attribution import (
    _pair_deltas,
    attribute_directional_confusions,
)
from coffee_detector.sni21_ontology import load_sni21_ontology


def test_attribution_separates_primary_family_and_cross_family_errors() -> None:
    confusion = {
        "biji_hitam": {
            "biji_hitam_sebagian": 3,
            "biji_muda": 2,
            "kulit_kopi_ukuran_besar": 1,
            "biji_hitam": 9,
        }
    }
    result = attribute_directional_confusions(confusion, load_sni21_ontology())
    assert result["wrong_class"] == 6
    assert result["category_counts"] == {
        "same_primary_condition": 3,
        "same_entity_family_only": 2,
        "cross_entity_family": 1,
    }
    assert result["same_task_counts"]["primary_condition"] == 3
    assert result["same_task_counts"]["entity_family"] == 5


def test_pair_deltas_rank_increased_errors_first() -> None:
    baseline = [{"expected": "a", "predicted": "b", "count": 2}]
    candidate = [
        {"expected": "a", "predicted": "b", "count": 5},
        {"expected": "c", "predicted": "d", "count": 1},
    ]
    rows = _pair_deltas(candidate, baseline)
    assert rows[0]["delta"] == 3
    assert rows[1]["delta"] == 1
