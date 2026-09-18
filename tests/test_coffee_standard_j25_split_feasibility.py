import pytest

from coffee_detector.analysis.coffee_standard_j25_split_feasibility import (
    RETRACTION,
    audit_j25_split_feasibility,
)


def test_retracted_filename_identity_audit_fails_fast() -> None:
    with pytest.raises(RuntimeError, match="RETRACTED"):
        audit_j25_split_feasibility("a", "b", "c", "d")
    assert "451 source images" in RETRACTION
