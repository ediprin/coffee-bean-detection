import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks/Faruq_V3_ACMC_Adrian_External_Colab.ipynb"
)


def test_notebook_uses_frozen_checkpoints_and_never_trains_or_opens_test() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "restore_real_a0_validation" in source
    assert "A0_real.tar" in source
    assert "faruq_grouped_manifest.json" in source
    assert "faruq_v3_acmc_locked_test_summary.json" in source
    assert "run_faruq_v3_acmc_adrian_external" in source
    assert "D0FT_seed42" in source and "ACMC1_seed42" in source
    assert "D0FT_seed123" in source and "ACMC1_seed123" in source
    assert "D0FT_seed2026" in source and "ACMC1_seed2026" in source
    assert "--seeds', '42', '123', '2026'" in source
    assert "--authorize-training" not in source
    assert "test_images_accessed" in source
    assert "locked-test NOT_CONFIRMED" in source
