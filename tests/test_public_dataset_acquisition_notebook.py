import json
from pathlib import Path


NOTEBOOK = Path("notebooks/Public_Coffee_Dataset_Acquire_And_Audit_Colab.ipynb")


def test_public_dataset_acquisition_notebook_freezes_exact_versions_without_training() -> None:
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
    assert "userdata.get('ROBOFLOW_API_KEY')" in source
    assert "'capstone_v1'" in source and "'version':1" in source
    assert "'lulus_v1'" in source and "'niacubilla_v1'" in source
    assert ".download('yolov8'" in source
    assert "archive_sha256" in source
    assert "v2_acquired_registry.yaml" in source
    assert "audit_public_dataset_registry" in source
    assert "model.train" not in source
    assert "YOLO(" not in source
    assert "best.pt" not in source
