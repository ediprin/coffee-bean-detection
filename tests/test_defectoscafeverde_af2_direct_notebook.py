import json
from pathlib import Path


NOTEBOOK = Path("notebooks/DefectosCafeVerde_AF2_Direct_Seed42_Colab.ipynb")


def _source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_notebook_uses_frozen_archive_and_branch() -> None:
    source = _source()
    assert "codex/public-dataset-eligibility-audit" in source
    assert "defectoscafeverde-grouped-physical-v1.tar" in source
    assert "ARCHIVE REASSEMBLED FROM" in source
    assert "range(11)" in source
    assert "53fb2233f1f0d1c77cb24eca2d720f86e0a16835b8a69f4e8f3176fae1aacef2" in source


def test_notebook_extracts_development_only() -> None:
    source = _source()
    assert "first in {'train','val'}" in source
    assert "TEST TEREXPOSE" in source
    assert "payload.pop('test',None)" in source
    assert "--grouped-audit" in source


def test_notebook_runs_matched_seed42_pair_and_preserves_test_lock() -> None:
    source = _source()
    assert "run_defectoscafeverde_af2_direct" in source
    assert "--seed','42'" in source
    assert "--authorize-training" in source
    assert "D0DIRECT" in source and "AF2DIRECT" in source
    assert "Jangan membuka test." in source
