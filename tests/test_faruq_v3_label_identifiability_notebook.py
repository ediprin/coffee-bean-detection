import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Faruq_V3_Label_Identifiability_Audit_Colab.ipynb"


def test_identifiability_notebook_is_training_free_and_test_locked() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "audit_faruq_v3_label_identifiability" in source
    assert "D0_seed42_diagnostic.json" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "model.train" not in source
    assert "RUN_SCREEN" not in source
    assert "force_remount=True" in source
    assert "sys.path.insert" in source
