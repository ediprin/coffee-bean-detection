from __future__ import annotations

import json
from pathlib import Path

from coffee_detector.analysis.af2cue1_parent_bootstrap import paired_parent_bootstrap


def _observation(name: str, class_id: int, *, correct: bool) -> dict:
    target = {"class_id": class_id, "xyxy": [0.0, 0.0, 10.0, 10.0]}
    predicted_box = [0.0, 0.0, 10.0, 10.0] if correct else [20.0, 20.0, 30.0, 30.0]
    prediction = {"class_id": class_id, "score": 0.9, "xyxy": predicted_box}
    return {"image_name": name, "targets": [target], "predictions": [prediction]}


def test_parent_bootstrap_resamples_clusters_and_detects_cue_gain() -> None:
    rows = {arm: [] for arm in ("AF2BASE", "AF2SPDS", "AF2CUE1")}
    parent_by_image = {}
    for class_id in range(21):
        for parent_index in range(5):
            name = f"class{class_id:02d}_parent{parent_index}.jpg"
            parent_by_image[name] = f"parent-{class_id:02d}-{parent_index}"
            rows["AF2BASE"].append(_observation(name, class_id, correct=False))
            rows["AF2SPDS"].append(_observation(name, class_id, correct=False))
            rows["AF2CUE1"].append(_observation(name, class_id, correct=True))

    result = paired_parent_bootstrap(
        rows, parent_by_image, iterations=20, seed=7, class_count=21
    )
    assert result["unit"] == "source_parent_cluster"
    assert result["independent_parents"] == 105
    assert result["validation_images"] == 105
    for comparison in result["comparisons"].values():
        for metric in comparison.values():
            assert metric["point_delta"] > 0.9
            assert metric["probability_positive"] == 1.0


def test_parent_bootstrap_colab_compiles_and_has_no_training_or_test() -> None:
    path = Path("notebooks/Faruq_V3_AF2CUE1_Parent_Bootstrap_Colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path}:{index}", "exec")
    assert "run_af2cue1_parent_bootstrap" in code
    assert "assert not (DATA/'test').exists()" in code
    assert "authorize-training" not in code
    assert "model.train" not in code
    assert "split='test'" not in code
