import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Faruq_V3_Predicted_ROI_Transfer_Colab.ipynb"


def test_predicted_roi_notebook_uses_frozen_d0_and_locks_test() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "run_faruq_v3_predicted_roi_transfer" in source
    assert "D0_seed42/weights/best.pt" in source
    assert "pyramid_separability.json" in source
    assert "candidate_count=500" in source
    assert "pca_components=128" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "detector_training_executed" in source
    assert "model.train(" not in source
    assert "sys.modules.pop(module_name, None)" in source
