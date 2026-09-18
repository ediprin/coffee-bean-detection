import pytest

from coffee_detector.analysis.coffee_standard_j25_visual_audit import (
    RETRACTION,
    audit_coffee_standard_j25_visuals,
)


def test_grouped_visual_audit_is_retracted() -> None:
    with pytest.raises(RuntimeError, match="RETRACTED"):
        audit_coffee_standard_j25_visuals("grouped", "source", "output")
    assert "official author split" in RETRACTION
