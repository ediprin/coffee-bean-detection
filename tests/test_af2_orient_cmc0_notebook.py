import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Faruq_V3_AF2_ORIENT_CMC0_Seed42_Colab.ipynb"


def _source_text() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )


def test_notebook_is_valid_and_pins_exact_branch():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    text = _source_text()
    assert "agent/af2-orient-cmc0-screening" in text
    assert "ultralytics.__version__ == '8.4.96'" in text


def test_notebook_runs_static_then_authorized_training_with_matched_parent():
    text = _source_text()
    assert "--stage','static'" in text
    assert "--stage','train'" in text
    assert "--d0-checkpoint" in text
    assert "--af2-orient-result" in text
    assert "--authorize-training" in text
    assert "AF2_ORIENT_seed42_result.json" in text
    assert "run_faruq_v3_af2_cpe_seed42_worker" not in text


def test_notebook_uses_clean_v2_output_and_keeps_test_locked():
    text = _source_text()
    assert "faruq-v3-af2-orient-cmc0-seed42-v2" in text
    assert "assert not (DATA_ROOT/'test').exists()" in text
    assert "split','test'" not in text
    assert 'split="test"' not in text
