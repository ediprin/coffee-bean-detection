import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Faruq_V3_Fusion_CM512_Colab.ipynb"


def test_cm512_notebook_is_cache_only_and_fixed_rank() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "run_faruq_v3_fusion_cm512" in source
    assert "predicted_roi_cache_train.npz" in source
    assert "predicted_roi_cache_val.npz" in source
    assert "components=512" in source
    assert "detector_inference_executed" in source
    assert "detector_training_executed" in source
    assert "model.train(" not in source
    assert "candidate_count" not in source
    assert "sys.modules.pop(module_name, None)" in source
