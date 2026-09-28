import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/DefectosCafeVerde_LIFRPF1_Seed42_Colab.ipynb"


def test_notebook_is_resumable_fail_fast_and_test_locked():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/defectoscafeverde-lif-rpf'" in code
    assert "run_defectoscafeverde_lif_rpf" in code
    assert "--authorize-training" in code
    assert "last.pt tersimpan di Drive setiap epoch" in code
    assert "first in {'train','val'}" in code
    assert "EXPECTED_ARCHIVE_BYTES=686474752" in code
    assert "('part16',41),('chunk',11),('part',2)" in code
    assert "fallback Drive API folder bersama" in code
    assert "MediaIoBaseDownload" in code
    assert code.index("ARCHIVE VALID:") < code.index("git','clone")
    assert code.index("D0 REFERENCE VALID:") < code.index("START/RESUME:")
    assert "test/images" not in code
    assert "--authorize-test" not in code
