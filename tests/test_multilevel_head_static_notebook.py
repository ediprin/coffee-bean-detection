import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Faruq_V3_Multilevel_Head_Static_Audit_Colab.ipynb"


def test_static_notebook_does_not_train_or_access_dataset() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "static_multilevel_head_audit" in source
    assert "fusion_cm512.json" in source
    assert "dataset_accessed" in source
    assert "test_images_accessed" in source
    assert "model.train(" not in source
    assert "subprocess.run(command" not in source
    assert "sys.modules.pop(module_name, None)" in source
