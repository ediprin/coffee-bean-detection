import json
from pathlib import Path


NOTEBOOK = Path("notebooks/Public_Coffee_Dataset_Eligibility_Audit_Colab.ipynb")


def test_public_dataset_audit_notebook_is_cpu_only_and_training_free() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(
        line
        for cell in payload["cells"]
        for line in cell.get("source", [])
    )

    assert payload["metadata"]["accelerator"] == "CPU"
    assert "pip','install','-q','-e'" in source
    assert "audit_public_dataset_registry" in source
    assert "model.train" not in source
    assert "YOLO(" not in source
    assert "best.pt" not in source
    assert "authorize-test" not in source
    assert "TRAINING AUTHORIZED:" in source
