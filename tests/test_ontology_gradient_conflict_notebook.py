import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Faruq_V3_Gradient_Conflict_Audit_Colab.ipynb"


def test_gradient_audit_notebook_is_train_only_and_does_not_train() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "run_ontology_gradient_conflict_audit" in source
    assert "D0_seed42/weights/best.pt" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "model.train(" not in source
    assert "sys.modules.pop(module_name, None)" in source
    assert "--max-batches" not in source
    assert "24" in source and "batch_size=8" in source
