import json
from pathlib import Path

import pytest
import torch
import yaml

from coffee_detector.defectos_dual_view.model import (
    DualViewFusionConfig,
    SymmetricDualViewFuser,
)
from coffee_detector.experiments.run_defectoscafeverde_dual_view_fusion import (
    _split_samples,
)
from coffee_detector.experiments.run_defectoscafeverde_dvsr1_locked_test import (
    _fuser_correct,
    run_locked_test,
)


def test_test_split_requires_explicit_locked_runner_but_is_parseable(tmp_path: Path):
    names = {index: f"class-{index}" for index in range(12)}
    (tmp_path / "test/images").mkdir(parents=True)
    (tmp_path / "test/labels").mkdir(parents=True)
    (tmp_path / "test/images/sample.jpg").write_bytes(b"placeholder")
    (tmp_path / "test/labels/sample.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (tmp_path / "data.yaml").write_text(
        yaml.safe_dump({"names": names}), encoding="utf-8"
    )

    parsed_names, samples = _split_samples(tmp_path, "test")

    assert parsed_names == names
    assert len(samples) == 1
    with pytest.raises(PermissionError, match="authorize-test"):
        run_locked_test(*([tmp_path / "missing"] * 12))


def test_fuser_endpoint_uses_native_view_box_for_fused_class():
    config = DualViewFusionConfig(classes=12, hidden=24, residual_limit=2.0)
    fuser = SymmetricDualViewFuser(config).eval()
    logits = torch.full((2, 2, 12), -5.0)
    labels = torch.tensor([0, 1])
    logits[0, 0, 0] = 4.0
    logits[0, 1, 0] = 3.0
    logits[1, 0, 1] = 4.0
    logits[1, 1, 1] = 3.0
    cache = {
        "view_logits": logits,
        "labels": labels,
        "ious": torch.tensor([[0.9, 0.2], [0.9, 0.2]]),
    }

    correct = _fuser_correct(fuser, cache, torch.device("cpu"))

    assert correct.tolist() == [True, True]


def test_locked_test_protocol_and_colab_are_final_and_test_only():
    protocol = Path(
        "docs/DEFECTOSCAFEVERDE_DVSR1_LOCKED_TEST_PROTOCOL_2026-10-06.md"
    ).read_text(encoding="utf-8")
    notebook = json.loads(
        Path(
            "notebooks/DefectosCafeVerde_DVSR1_Locked_Test_Colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "before any grouped test image is extracted or accessed" in protocol
    assert "REPORT_LOCKED_TEST_AND_STOP" in protocol
    assert "No additional" in protocol
    compile(source, "dvsr1_locked_test_colab", "exec")
    assert "run_defectoscafeverde_dvsr1_locked_test" in source
    assert "--authorize-test" in source
    assert "first=='test'" in source
    assert "TEST:" in source
