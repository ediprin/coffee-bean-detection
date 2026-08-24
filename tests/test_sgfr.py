import json
from pathlib import Path

import torch
import yaml

from coffee_detector.experiments.run_faruq_v3_sgfr_synthesis import _comparison
from coffee_detector.sgfr import (
    SGFRConfig,
    SGFRDetectHead,
    SGFRTaskModel,
    load_sgfr_weights,
    static_sgfr_audit,
)
from coffee_detector.stb import STBConfig, STBDetectionModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIG_ROOT = ROOT / "configs/sgfr"


def _models(stage: str, nc: int = 5):
    torch.manual_seed(71)
    source = STBDetectionModel(
        str(MODEL_YAML), nc=nc, verbose=False, stb=STBConfig()
    ).eval()
    candidate = SGFRTaskModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        sgfr=SGFRConfig(stage=stage, frequency_hidden=8),
    ).eval()
    load_sgfr_weights(candidate, source)
    return source, candidate


def test_all_sgfr_stages_start_exactly_from_stb_and_keep_native_boxes():
    image = torch.rand(1, 3, 128, 128)
    for stage in ("control", "geometry", "frequency"):
        source, candidate = _models(stage)
        with torch.inference_mode():
            native = source(image)
            zero = candidate(image)
        head = candidate.model[-1]
        assert isinstance(head, SGFRDetectHead)
        assert torch.equal(zero[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"])
        assert torch.equal(zero[1]["one2one"]["scores"], native[1]["one2one"]["scores"])

        if stage == "geometry":
            with torch.no_grad():
                for level in head.geometry_levels:
                    level.class_correction.bias.fill_(0.1)
        elif stage == "frequency":
            with torch.no_grad():
                for level in head.frequency_levels:
                    level.class_correction.bias.fill_(0.1)
        with torch.inference_mode():
            active = candidate(image)
        assert torch.equal(active[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"])
        if stage == "control":
            assert torch.equal(active[1]["one2one"]["scores"], native[1]["one2one"]["scores"])
        else:
            assert not torch.equal(active[1]["one2one"]["scores"], native[1]["one2one"]["scores"])


def test_stage_freeze_policy_is_exact_and_forces_frozen_buffers_to_eval():
    expected = {
        "control": ("base_head", "cv3"),
        "geometry": ("geometry_levels",),
        "frequency": ("frequency_levels",),
    }
    for stage, fragments in expected.items():
        _source, candidate = _models(stage)
        policy = candidate.apply_freeze_policy()
        candidate.train(True)
        names = [name for name, value in candidate.named_parameters() if value.requires_grad]
        assert policy["trainable"] > 0
        assert names and all(any(fragment in name for fragment in fragments) for name in names)
        assert all(not layer.training for layer in list(candidate.model)[:-1])
        assert not candidate.model[-1].blocks.training
        if stage != "control":
            assert not candidate.model[-1].base_head.training


def test_only_authorized_residual_receives_gradients_and_boxes_never_do():
    image = torch.rand(1, 3, 128, 128)
    for stage, authorized in (("geometry", "geometry_levels"), ("frequency", "frequency_levels")):
        _source, candidate = _models(stage)
        candidate.apply_freeze_policy()
        candidate.train(True)
        scores = candidate(image)["one2many"]["scores"]
        scores.square().mean().backward()
        gradients = [
            name
            for name, value in candidate.named_parameters()
            if value.grad is not None
        ]
        assert gradients and all(authorized in name for name in gradients)
        assert all(
            value.grad is None
            for value in candidate.model[-1].one2many["box_head"].parameters()
        )


def test_geometry_and_frequency_run_native_yolo26_loss():
    batch = {
        "img": torch.rand(1, 3, 128, 128),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    for stage in ("geometry", "frequency"):
        _source, candidate = _models(stage)
        candidate.args = type(
            "Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 2}
        )()
        candidate.apply_freeze_policy()
        candidate.train(True)
        loss, components = candidate(batch)
        assert torch.isfinite(loss).all()
        assert torch.isfinite(components).all()
        loss.sum().backward()


def test_three_configs_share_schema_and_freeze_short_stage_schedule():
    expected = {"SGC0": "control", "SGI1": "geometry", "SGF2": "frequency"}
    files = sorted(CONFIG_ROOT.glob("*.yaml"))
    assert len(files) == 3
    payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in files]
    for payload in payloads:
        assert payload["sgfr"]["stage"] == expected[payload["code"]]
        assert payload["train"]["epochs"] == 20
        assert payload["train"]["imgsz"] == 640
        assert payload["train"]["batch"] == 16
        assert payload["train"]["pretrained"] is False
    keys = [set(payload["sgfr"]) for payload in payloads]
    assert keys[0] == keys[1] == keys[2]


def test_static_audit_passes_with_real_stb_checkpoint_schema(tmp_path: Path):
    source = STBDetectionModel(
        str(MODEL_YAML), nc=5, verbose=False, stb=STBConfig()
    )
    checkpoint = tmp_path / "stb.pt"
    torch.save({"model": source}, checkpoint)
    result = static_sgfr_audit(
        MODEL_YAML, checkpoint, tmp_path / "static.json", nc=5, image_size=64
    )
    assert result["decision"] == "PASS"
    assert all(value["decision"] == "PASS" for value in result["arms"].values())
    assert result["arms"]["SGF2"]["trainable_parameters"] < result["arms"]["SGI1"]["trainable_parameters"]


def test_notebook_is_branch_correct_resumable_staged_and_val_only():
    notebook = ROOT / "notebooks/Faruq_V3_SGFR_Frozen_Synthesis_Colab.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "agent/stb-igem-af2-frozen-synthesis" in source
    assert "resolve_drive_project_root(required_relative_paths=REQUIRED)" in source
    assert "run_faruq_v3_sgfr_synthesis" in source
    assert "'--stage','geometry'" in source
    assert "'--stage','frequency'" in source
    assert "--authorize-training" in source
    assert "split=test" not in source.lower()


def test_protocol_is_frozen_and_forbids_capacity_free_claim():
    protocol = (
        ROOT / "docs/FARUQ_V3_SGFR_FROZEN_SYNTHESIS_PROTOCOL.md"
    ).read_text(encoding="utf-8")
    assert "Status: frozen before training" in protocol
    assert "SGC0" in protocol and "SGI1" in protocol and "SGF2" in protocol
    assert "Test must not be extracted" in protocol
    assert "gain is not" in protocol.lower() and "capacity-free" in protocol
    assert "7,331,021" in protocol


def test_sgfr_gate_requires_macro_gain_and_lower_tail_preservation():
    reference = {
        "macro_map50_95": 0.88,
        "bottom3_class_map50_95": 0.82,
        "worst_class_map50_95": 0.80,
    }
    passing = {
        "macro_map50_95": 0.886,
        "bottom3_class_map50_95": 0.83,
        "worst_class_map50_95": 0.795,
    }
    failing = dict(passing, bottom3_class_map50_95=0.81)
    assert _comparison(passing, reference)["decision"] == "PASS"
    assert _comparison(failing, reference)["decision"] == "FAIL"
