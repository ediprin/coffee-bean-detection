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
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), str(NOTEBOOK), "exec")

    assert payload["metadata"]["accelerator"] == "CPU"
    assert "pip','install','-q','-e'" in source
    assert "REMOTE='https://'+'github.com/ediprin/coffee-bean-detection.git'" in source
    assert "for attempt in range(1,4)" in source
    assert "capture_output=True" in source
    assert "audit_public_dataset_registry" in source
    assert "model.train" not in source
    assert "YOLO(" not in source
    assert "best.pt" not in source
    assert "authorize-test" not in source
    assert "TRAINING AUTHORIZED:" in source
