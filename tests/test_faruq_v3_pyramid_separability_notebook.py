import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Faruq_V3_Pyramid_Separability_Colab.ipynb"


def test_pyramid_notebook_freezes_detector_and_locks_test() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "run_faruq_v3_pyramid_separability" in source
    assert "D0_seed42/weights/best.pt" in source
    assert "faruq-development-v3-grouped.tar" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "detector_training_executed" in source
    assert "probe_fitting_executed" in source
    assert "model.train(" not in source
    assert "sys.modules.pop(module_name, None)" in source
    assert "feature_cache" not in source  # runner owns resumable cache paths
