import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_parallel_notebooks_are_resumable_and_test_locked():
    for arm in ("D0DIRECT", "DCWCF1"):
        path = ROOT / f"notebooks/DefectosCafeVerde_{arm}_Seed42_Colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        ast.parse(code)
        assert f"ARM='{arm}'" in code
        assert "BRANCH='codex/defectoscafeverde-cwcf-direct'" in code
        assert "run_defectoscafeverde_cwcf_direct" in code
        assert "--authorize-training" in code
        assert "last.pt tersimpan di Drive setiap epoch" in code
        assert "first in {'train','val'}" in code
        assert "test/images" not in code
        assert "--authorize-test" not in code
