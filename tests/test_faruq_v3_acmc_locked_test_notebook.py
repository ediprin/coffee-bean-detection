import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks/Faruq_V3_ACMC_Locked_Test_Colab.ipynb"


def test_locked_test_notebook_is_integrity_gated_resumable_and_training_free() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "prepare_faruq_locked_test" in source
    assert "faruq_locked_test_eligibility.json" in source
    assert "if eligibility['decision'] != 'PASS'" in source
    assert "--authorize-test" in source
    assert "run_faruq_v3_acmc_locked_test" in source
    assert "LOCKED_ARCHIVE.is_file()" in source
    assert "ROBOFLOW_API_KEY" in source
    assert "train_experiment" not in source
    assert "--authorize-training" not in source
    assert "test sudah dibuka; jangan tuning atau training lagi" in source.lower()
