import json
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from coffee_detector.experiments.run_faruq_v3_af2_uniform_soup import (
    _decision,
    _validate_confirmation,
    average_state_dicts,
    build_uniform_af2_soup,
)


class _FakeAF2Config:
    mode = "af2"

    def to_dict(self):
        return {"mode": self.mode, "patch_size": 32}


class _FakeAF2Model(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value]))
        self.register_buffer("structural", torch.tensor(9))
        self.afab_config = _FakeAF2Config()
        self.yaml = {"backbone": ["same"]}
        self.names = {0: "bean"}


def test_uniform_average_uses_float64_and_preserves_structural_buffers() -> None:
    states = [
        OrderedDict(weight=torch.tensor([value], dtype=torch.float16), count=torch.tensor(7))
        for value in (1.0, 2.0, 6.0)
    ]
    averaged, audit = average_state_dicts(states)
    assert averaged["weight"].dtype == torch.float16
    assert averaged["weight"].item() == pytest.approx(3.0, abs=1e-3)
    assert averaged["count"].item() == 7
    assert audit == {
        "state_tensors": 2,
        "averaged_floating_tensors": 1,
        "preserved_structural_tensors": 1,
    }


def test_average_rejects_schema_dtype_and_structural_mismatch() -> None:
    good = OrderedDict(weight=torch.ones(2), count=torch.tensor(1))
    with pytest.raises(RuntimeError, match="schema"):
        average_state_dicts([good, OrderedDict(other=torch.ones(2)), good])
    with pytest.raises(RuntimeError, match="Dtype"):
        average_state_dicts(
            [good, OrderedDict(weight=torch.ones(2, dtype=torch.float64), count=torch.tensor(1)), good]
        )
    with pytest.raises(RuntimeError, match="struktural"):
        average_state_dicts(
            [good, OrderedDict(weight=torch.ones(2), count=torch.tensor(2)), good]
        )


def test_build_soup_persists_uniform_average_and_audit(tmp_path: Path) -> None:
    sources = []
    for seed, value in zip((42, 123, 2026), (1.0, 2.0, 6.0)):
        path = tmp_path / f"seed{seed}.pt"
        torch.save({"model": _FakeAF2Model(value), "ema": None, "train_args": {"seed": seed}}, path)
        sources.append(path)
    output = tmp_path / "soup.pt"
    audit = build_uniform_af2_soup(sources, output)
    assert audit["decision"] == "PASS"
    assert audit["gates"]["checkpoint_seeds_exact"]
    from ultralytics.utils.patches import torch_load

    model = torch_load(output, map_location="cpu")["model"]
    assert model.weight.float().item() == pytest.approx(3.0, abs=1e-3)
    assert model.structural.item() == 9
    assert torch_load(output, map_location="cpu")["optimizer"] is None


def test_confirmation_and_tail_stabilization_gate(tmp_path: Path) -> None:
    per_seed = {
        str(seed): {
            "AF2": {
                "macro_map50_95": macro,
                "bottom3_class_map50_95": bottom,
                "worst_class_map50_95": worst,
            }
        }
        for seed, macro, bottom, worst in (
            (42, 0.88, 0.79, 0.77),
            (123, 0.87, 0.78, 0.76),
            (2026, 0.89, 0.80, 0.78),
        )
    }
    payload = {
        "protocol": "faruq-v3-af2-igem-paired-validation-confirmation-v1",
        "seeds": [42, 123, 2026],
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "decisions": {"AF2": {"decision": "PASS"}},
        "per_seed": per_seed,
    }
    path = tmp_path / "confirmation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _, reference = _validate_confirmation(path)
    comparison, decision = _decision(
        {
            "macro_map50_95": reference["macro_map50_95"] + 0.001,
            "bottom3_class_map50_95": reference["bottom3_class_map50_95"] + 0.005,
            "worst_class_map50_95": reference["worst_class_map50_95"] + 0.001,
        },
        reference,
    )
    assert decision == "RETAIN"
    assert all(comparison["criteria"].values())


def test_notebook_is_validation_only_and_uses_exact_checkpoints() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks/Faruq_V3_AF2_Uniform_Model_Soup_Colab.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "codex/af2-uniform-model-soup" in source
    assert "AF2_seed42/weights/best.pt" in source
    assert "AF2_seed123/weights/best.pt" in source
    assert "AF2_seed2026/weights/best.pt" in source
    assert "--confirmation-summary" in source
    assert "--checkpoints" in source
    assert "--authorize-training" not in source
    assert "split=test" not in source.lower()
    assert "test tidak" in source.lower()
