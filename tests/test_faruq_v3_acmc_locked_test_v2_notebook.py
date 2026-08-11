import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks/Faruq_V3_ACMC_Locked_Test_V2_Colab.ipynb"


def test_v2_notebook_freezes_amendment_before_inference_and_never_trains() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    amendment = source.index("prepare_faruq_locked_test_amendment")
    inference = source.index("run_faruq_v3_acmc_locked_test")
    assert amendment < inference
    assert "--amendment-summary" in source
    assert "--authorize-test" in source
    assert "faruq-v3-locked-test-v1.tar" in source
    assert "model_inference_executed" in source
    assert "train_experiment" not in source
    assert "--authorize-training" not in source
    assert "tidak ada tuning berikutnya" in source
