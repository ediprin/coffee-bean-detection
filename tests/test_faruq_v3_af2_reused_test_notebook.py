import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks/Faruq_V3_AF2_Reused_Test_Posthoc_Colab.ipynb"
)


def test_notebook_uses_exact_artifacts_and_never_trains() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "agent/af2-continuation-confirmation" in source
    assert "faruq-v3-locked-test-v1.tar" in source
    assert "D0FT_seed42_test.json" in source
    assert "D0FT_seed123_test.json" in source
    assert "D0FT_seed2026_test.json" in source
    assert "AF2_seed42/weights/best.pt" in source
    assert "AF2_seed123/weights/best.pt" in source
    assert "AF2_seed2026/weights/best.pt" in source
    assert "run_faruq_v3_af2_reused_test" in source
    assert "--authorize-reused-test" in source
    assert "--authorize-training" not in source
    assert "train_experiment" not in source
    assert "REUSED_TEST_POSTHOC_NOT_LOCKED_CONFIRMATION" in source
    assert "os.chdir('/content')" in source
