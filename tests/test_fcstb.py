import json
from pathlib import Path

import torch
import yaml

from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.afab.operator import AFABConfig
from coffee_detector.experiments.run_faruq_v3_fcstb import _comparison
from coffee_detector.fcstb import (
    FCSTBConfig,
    FCSTBTaskModel,
    gt_bounded_logit_distillation,
    load_fcstb_weights,
    static_fcstb_audit,
)
from coffee_detector.stb import STBConfig, STBDetectionModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _student(mode="control", teacher=None, nc=5):
    source = STBDetectionModel(
        str(MODEL_YAML), nc=nc, verbose=False, stb=STBConfig()
    ).eval()
    config = FCSTBConfig(mode=mode, teacher_checkpoint=str(teacher) if teacher else None)
    candidate = FCSTBTaskModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        stb=STBConfig().to_dict(),
        fcstb=config,
    ).eval()
    load_fcstb_weights(candidate, source)
    return source, candidate


def test_fcstb_student_starts_bitwise_as_stb_and_has_no_teacher_module():
    image = torch.rand(1, 3, 128, 128)
    for mode, teacher in (("control", None), ("distill", "teacher.pt")):
        source, candidate = _student(mode, teacher)
        with torch.inference_mode():
            left, right = source(image), candidate(image)
        assert torch.equal(left[0], right[0])
        assert torch.equal(left[1]["one2one"]["boxes"], right[1]["one2one"]["boxes"])
        assert torch.equal(left[1]["one2one"]["scores"], right[1]["one2one"]["scores"])
        assert source.state_dict().keys() == candidate.state_dict().keys()
        assert not any("teacher" in name for name, _ in candidate.named_modules())


def test_fcstb_freeze_policy_allows_only_stb_and_classification_heads():
    _source, candidate = _student()
    policy = candidate.apply_freeze_policy()
    candidate.train(True)
    names = [name for name, value in candidate.named_parameters() if value.requires_grad]
    assert policy["trainable"] > 0 and names
    assert all(
        ".blocks." in name
        or (".base_head." in name and (".cv3." in name or ".one2one_cv3." in name))
        for name in names
    )
    assert all(not layer.training for layer in list(candidate.model)[:-1])
    head = candidate.model[-1]
    assert all(
        not module.training
        for branch in (head.one2many, head.one2one)
        for module in branch["box_head"]
    )


def test_gt_bounded_distillation_uses_only_correct_confident_teacher_rows():
    student = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 3.0, -1.0], [3.0, 0.0, -1.0]])
    labels = torch.tensor([1, 1])
    loss, stats = gt_bounded_logit_distillation(
        student,
        teacher,
        labels,
        temperature=2.0,
        minimum_gt_probability=0.1,
    )
    assert stats["positive_anchors"] == 2
    assert stats["teacher_correct_anchors"] == 1
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad[1]) == 0


def test_distillation_task_runs_native_yolo_loss_with_training_only_teacher(tmp_path):
    teacher = AFABDetectionModel(
        str(MODEL_YAML), nc=5, verbose=False, afab=AFABConfig(mode="af2")
    )
    teacher_path = tmp_path / "af2.pt"
    torch.save({"model": teacher}, teacher_path)
    _source, candidate = _student("distill", teacher_path)
    candidate.args = type(
        "Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 2}
    )()
    candidate.apply_freeze_policy()
    candidate.train(True)
    batch = {
        "img": torch.rand(1, 3, 128, 128),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    loss, components = candidate(batch)
    assert torch.isfinite(loss).all() and torch.isfinite(components).all()
    loss.sum().backward()
    assert all(
        parameter.grad is None
        for parameter in candidate.model[-1].one2one["box_head"].parameters()
    )


def test_static_audit_passes_and_students_are_capacity_matched(tmp_path):
    source = STBDetectionModel(
        str(MODEL_YAML), nc=5, verbose=False, stb=STBConfig()
    )
    stb = tmp_path / "stb.pt"
    af2 = tmp_path / "af2.pt"
    torch.save({"model": source}, stb)
    af2.write_bytes(b"teacher-checkpoint-hash-only")
    result = static_fcstb_audit(
        MODEL_YAML, stb, af2, tmp_path / "static.json", nc=5, image_size=64
    )
    assert result["decision"] == "PASS"
    assert result["arms"]["FCT0"]["parameters"] == result["arms"]["FCD1"]["parameters"]
    assert result["arms"]["FCT0"]["trainable_parameters"] == result["arms"]["FCD1"]["trainable_parameters"]


def test_fcstb_configs_are_schedule_and_capacity_matched():
    files = sorted((ROOT / "configs/fcstb").glob("*.yaml"))
    assert len(files) == 2
    rows = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in files]
    assert {row["code"] for row in rows} == {"FCT0", "FCD1"}
    assert rows[0]["stb"] == rows[1]["stb"]
    assert rows[0]["train"] == rows[1]["train"]
    assert all(row["train"]["epochs"] == 20 for row in rows)
    assert all(row["train"]["pretrained"] is False for row in rows)


def test_gate_requires_distillation_to_beat_stb_and_matched_control():
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
    assert _comparison(passing, reference)["decision"] == "PASS"
    assert _comparison(dict(passing, macro_map50_95=0.884), reference)["decision"] == "FAIL"


def test_protocol_and_notebook_are_fail_fast_resumable_and_val_only():
    protocol = (ROOT / "docs/FARUQ_V3_FCSTB_DISTILLATION_PROTOCOL.md").read_text()
    assert "Status: frozen before training" in protocol
    assert "FCT0" in protocol and "FCD1" in protocol
    assert "Test\nmust not be extracted" in protocol
    notebook_path = ROOT / "notebooks/Faruq_V3_FCSTB_Distillation_Colab.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "agent/fc-stb-frequency-distillation" in source
    assert "resolve_drive_project_root(required_relative_paths=REQUIRED)" in source
    assert "'--stage','static'" in source
    assert "'--stage','diagnostic'" in source
    assert "'--stage','train'" in source
    assert "--authorize-training" in source
    assert "split=test" not in source.lower()
