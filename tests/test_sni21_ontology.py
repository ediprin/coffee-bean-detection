from pathlib import Path

import pytest

from coffee_detector.sni21_ontology import (
    SNI21_CLASSES,
    load_sni21_ontology,
    structured_target_for,
    validate_sni21_ontology,
)


def test_structured_ontology_is_complete_unique_and_protocol_only() -> None:
    ontology = load_sni21_ontology()
    assert tuple(ontology["classes"]) == SNI21_CLASSES
    assert ontology["status"] == "protocol_only_no_training_authorized"
    assert ontology["observability"]["physical_size_mm"] == "calibrated_scale_required"


def test_compound_and_size_targets_follow_sni_definitions() -> None:
    black_broken = structured_target_for("biji_hitam_pecah")
    assert black_broken["positive_flags"] == ("black", "broken")
    assert black_broken["observed_attributes"]["integrity_fraction"] == (
        "at_most_three_quarters"
    )

    parchment = structured_target_for("kulit_tanduk_ukuran_kecil")
    assert parchment["observed_attributes"]["relative_completeness"] == (
        "less_than_half"
    )
    assert "physical_size_mm" not in parchment["observed_attributes"]

    foreign = structured_target_for("tanah_batu_ranting_besar")
    assert foreign["observed_attributes"]["physical_size_mm"] == "greater_than_10"


def test_validation_rejects_training_authorization_and_bad_weights() -> None:
    ontology = load_sni21_ontology()
    ontology["status"] = "training"
    with pytest.raises(ValueError, match="protocol-only"):
        validate_sni21_ontology(ontology)

    ontology = load_sni21_ontology()
    ontology["classes"]["biji_hitam"]["defect_weight"] = 0.5
    with pytest.raises(ValueError, match="Bobot SNI"):
        validate_sni21_ontology(ontology)


def test_default_config_is_inside_repository() -> None:
    source = Path(load_sni21_ontology()["source"])
    assert source.name == "structured_ontology_v1.yaml"
    assert source.is_file()
