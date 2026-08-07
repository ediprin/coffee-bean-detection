import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/SNI21_Ontology_Marginal_Static_Audit_Colab.ipynb"


def test_static_notebook_has_no_dataset_training_or_test_access() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "audit_ontology_marginal_static" in source
    assert "force_remount=True" in source
    assert "sys.path.insert" in source
    assert "model.train" not in source
    assert "DATA_ROOT" not in source
    assert "test/images" not in source
