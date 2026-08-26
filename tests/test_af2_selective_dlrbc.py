from __future__ import annotations

import json
import ast
from pathlib import Path

import torch

from coffee_detector.af2_selective_dlrbc.model import (
    AF2SelectiveDLRBCConfig,
    SelectiveLowRankResidual,
)
from coffee_detector.analysis.af2_dlrbc_complementarity import (
    build_af2_dlrbc_complementarity_audit,
)


def test_selective_residual_zero_init_and_class_isolation():
    config = AF2SelectiveDLRBCConfig(selected_class_ids=(2, 7), rank=8)
    module = SelectiveLowRankResidual(32, 21, config)
    value = torch.randn(2, 32, 8, 8, requires_grad=True)
    zero = module(value)
    assert torch.equal(zero, torch.zeros_like(zero))

    with torch.no_grad():
        module.gate[[2, 7]] = 0.5
    active = module(value)
    unselected = [index for index in range(21) if index not in {2, 7}]
    assert torch.equal(active[:, unselected], torch.zeros_like(active[:, unselected]))
    assert active[:, [2, 7]].abs().sum().item() > 0.0
    active.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert module.gate.grad is not None and torch.isfinite(module.gate.grad).all()


def test_complementarity_audit_is_train_only(tmp_path):
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    names = {index: f"class_{index}" for index in range(21)}
    (root / "data.yaml").write_text(
        "names:\n" + "\n".join(f"  {index}: {name}" for index, name in names.items()) + "\n",
        encoding="utf-8",
    )
    af2_values = {name: 0.8 for name in names.values()}
    dlrbc_values = dict(af2_values)
    dlrbc_values["class_2"] = 0.84
    dlrbc_values["class_7"] = 0.83
    reports = []
    for label, values in (("af2", af2_values), ("dlrbc", dlrbc_values)):
        path = tmp_path / f"{label}.json"
        path.write_text(
            json.dumps(
                {
                    "data": str(root),
                    "split": "train",
                    "metrics": {
                        "map50_95_by_class": values,
                        "classes_without_ground_truth": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        reports.append(path)
    result = build_af2_dlrbc_complementarity_audit(
        root, reports[0], reports[1], tmp_path / "audit.json"
    )
    assert result["decision"] == "AUTHORIZE_AF2CSD1"
    assert result["selected_class_ids"] == [2, 7]
    assert result["validation_accessed"] is False
    assert result["test_images_accessed"] is False


def test_colab_notebook_compiles_and_preserves_split_lock():
    notebook_path = Path("notebooks/Faruq_V3_AF2_Class_Selective_DLRBC_Colab.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "codex/af2-selective-dlrbc" in source
    assert "--authorize-training" in source
    assert "split='train'" in source
    assert "assert not (DATA/'test').exists()" in source
    assert "last.pt" not in source or "RESUME" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"notebook:{index}")
