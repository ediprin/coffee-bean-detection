import json
from pathlib import Path


NOTEBOOK = Path("notebooks/Faruq_V3_AF2_Direct_From_Pretrained_Seed42_Kaggle.ipynb")


def _source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )


def test_direct_notebook_is_kaggle_seed42_and_uses_direct_branch():
    source = _source()
    assert "codex/af2-direct-from-pretrained" in source
    assert "Kaggle-only notebook" in source
    assert "--seed','42'" in source
    assert "--authorize-training" in source
    assert "coffee_detector.experiments.run_faruq_v3_af2_direct" in source


def test_direct_notebook_requires_development_archive_not_coffee_parent():
    source = _source()
    assert "faruq-development-v3-grouped.tar.bin" in source
    assert "YOLO('yolo26n.pt')" in source
    assert "D0_seed42_best.pt" not in source
    assert "--d0-checkpoint" not in source
    assert "--af2-checkpoint" not in source
    assert "--parent-result" not in source


def test_direct_notebook_preserves_test_lock_and_state_snapshot():
    source = _source()
    assert "TEST TEREXPOSE" in source
    assert "af2-direct-from-pretrained-seed42-state.zip" in source
    assert "Kirim tabel ini + SCREEN. Jangan buka test." in source
