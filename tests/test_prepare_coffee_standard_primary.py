import pytest

from coffee_detector.data.prepare_coffee_standard_primary import (
    RETRACTION,
    prepare_coffee_standard_primary,
)


def test_filename_grouped_primary_rebuild_is_retracted() -> None:
    with pytest.raises(RuntimeError, match="RETRACTED"):
        prepare_coffee_standard_primary("source", "output")
    assert "official split" in RETRACTION
