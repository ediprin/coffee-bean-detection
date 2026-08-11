import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks/Faruq_V3_Synthetic_Density_Colab.ipynb"


def test_notebook_is_validation_only_and_resumable() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "faruq-development-v3-grouped.tar" in source
    assert "faruq-development-v2.tar" in source
    assert "--polygon-root" in source
    assert "--library-cache-root" in source
    assert "run_faruq_v3_synthetic_density_setup" in source
    assert "run_faruq_v3_synthetic_density_screening" in source
    assert "D0FT_seed42" in source and "ACMC1_seed42" in source
    assert "--scenes-per-condition', '100'" in source
    assert "--authorize-training" not in source
    assert "test_images_accessed" in source
    assert "locked-test conclusion" in source
