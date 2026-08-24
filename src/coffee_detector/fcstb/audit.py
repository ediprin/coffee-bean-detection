from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from coffee_detector.stb.model import STBDetectHead

from .model import FCSTBConfig, FCSTBDetectionModel, load_fcstb_weights


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def _trainable_policy(model: FCSTBDetectionModel) -> bool:
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    return bool(names) and all(
        "model.23.blocks" in name
        or (
            "model.23.base_head" in name
            and (".cv3." in name or ".one2one_cv3." in name)
        )
        for name in names
    )


def static_fcstb_audit(
    model_yaml: str | Path,
    stb_checkpoint: str | Path,
    af2_checkpoint: str | Path,
    output: str | Path,
    *,
    nc: int = 21,
    image_size: int = 128,
) -> dict[str, Any]:
    from ultralytics import YOLO

    source_path = Path(stb_checkpoint).expanduser().resolve()
    teacher_path = Path(af2_checkpoint).expanduser().resolve()
    source = YOLO(str(source_path)).model.cpu().eval()
    if not isinstance(source.model[-1], STBDetectHead):
        raise TypeError("Static FC-STB memerlukan checkpoint STB")
    torch.manual_seed(42)
    image = torch.rand(1, 3, int(image_size), int(image_size))
    with torch.inference_mode():
        reference = source(image)
    arms = {}
    for code, mode in (("FCT0", "control"), ("FCD1", "distill")):
        config = FCSTBConfig(
            mode=mode,
            teacher_checkpoint=str(teacher_path) if mode == "distill" else None,
        )
        candidate = FCSTBDetectionModel(
            str(Path(model_yaml).resolve()),
            nc=nc,
            verbose=False,
            stb={"window_size": 4, "num_heads": 4, "mlp_ratio": 4.0},
            fcstb=config,
        ).cpu()
        transfer = load_fcstb_weights(candidate, source)
        candidate.eval()
        with torch.inference_mode():
            current = candidate(image)
        candidate.apply_freeze_policy()
        candidate.train(True)
        head = candidate.model[-1]
        gates = {
            "strict_stb_transfer": transfer["strict"] is True,
            "same_parameter_schema": candidate.state_dict().keys() == source.state_dict().keys(),
            "same_parameter_count": sum(p.numel() for p in candidate.parameters())
            == sum(p.numel() for p in source.parameters()),
            "zero_boxes_bitwise_equal": torch.equal(
                current[1]["one2one"]["boxes"], reference[1]["one2one"]["boxes"]
            ),
            "zero_scores_bitwise_equal": torch.equal(
                current[1]["one2one"]["scores"], reference[1]["one2one"]["scores"]
            ),
            "trainable_policy_exact": _trainable_policy(candidate),
            "backbone_neck_forced_eval": all(
                not layer.training for layer in list(candidate.model)[:-1]
            ),
            "box_heads_forced_eval": all(
                not module.training
                for branch in (head.one2many, head.one2one)
                for module in branch["box_head"]
            ),
            "teacher_not_registered_in_student": not any(
                "teacher" in name for name, _ in candidate.named_modules()
            ),
            "single_forward_inference": type(candidate).predict
            is FCSTBDetectionModel.predict,
        }
        arms[code] = {
            "mode": mode,
            "parameters": sum(p.numel() for p in candidate.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in candidate.parameters() if p.requires_grad
            ),
            "gates": gates,
            "decision": "PASS" if all(gates.values()) else "FAIL",
        }
    payload = {
        "protocol": "faruq-v3-fcstb-static-v1",
        "training_executed": False,
        "dataset_accessed": False,
        "test_images_accessed": False,
        "stb_sha256": _sha256(source_path),
        "af2_sha256": _sha256(teacher_path),
        "arms": arms,
        "decision": "PASS"
        if all(value["decision"] == "PASS" for value in arms.values())
        else "FAIL",
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(destination)
    return payload


def audit_fcstb_checkpoint_invariance(
    source_checkpoint: str | Path, candidate_checkpoint: str | Path
) -> dict[str, Any]:
    from ultralytics import YOLO

    source = YOLO(str(Path(source_checkpoint).resolve())).model.cpu().eval()
    candidate = YOLO(str(Path(candidate_checkpoint).resolve())).model.cpu().eval()
    source_head, candidate_head = source.model[-1], candidate.model[-1]
    if not isinstance(source_head, STBDetectHead) or not isinstance(
        candidate_head, STBDetectHead
    ):
        raise TypeError("Audit checkpoint FC-STB memerlukan dua STB head")
    gates = {
        "same_parameter_schema": source.state_dict().keys() == candidate.state_dict().keys(),
        "backbone_neck_unchanged": all(
            _equal(source.model[index], candidate.model[index])
            for index in range(len(source.model) - 1)
        ),
        "box_heads_unchanged": _equal(
            source_head.one2many["box_head"], candidate_head.one2many["box_head"]
        )
        and _equal(source_head.one2one["box_head"], candidate_head.one2one["box_head"]),
        "classification_path_changed": not _equal(source_head.blocks, candidate_head.blocks)
        or not _equal(
            source_head.one2one["cls_head"], candidate_head.one2one["cls_head"]
        ),
    }
    return {"gates": gates, "decision": "PASS" if all(gates.values()) else "FAIL"}
