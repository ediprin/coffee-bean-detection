from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from coffee_detector.af2_cpe import AF2CPEDetectionModel
from coffee_detector.fsce_cpe.loss import cpe_supervised_contrastive_loss
from coffee_detector.fsce_cpe.model import load_fsce_cpe_detector_weights

ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def run_static_audit(checkpoint: str | Path, output: str | Path, *, device: str = "cpu") -> dict:
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    configs = []
    for arm in ("AF2CPE0", "AF2CPE5"):
        path = ROOT / f"configs/af2_cpe/{arm}_yolo26n.yaml"
        configs.append(yaml.safe_load(path.read_text(encoding="utf-8")))
    left, right = configs
    left_same = {k: v for k, v in left["cpe"].items() if k != "loss_weight"}
    right_same = {k: v for k, v in right["cpe"].items() if k != "loss_weight"}
    config_matched = left["afab"] == right["afab"] and left["train"] == right["train"] and left_same == right_same

    from ultralytics import YOLO

    source = YOLO(str(checkpoint)).model
    models = []
    for payload in configs:
        model = AF2CPEDetectionModel(
            str(ROOT / payload["model"]), nc=int(source.nc), verbose=False,
            afab=payload["afab"], cpe=payload["cpe"],
        ).to(device)
        load_fsce_cpe_detector_weights(model, source)
        models.append(model)
    models[1].load_state_dict(models[0].state_dict(), strict=True)
    calls = [0, 0]
    hooks = []
    for index, model in enumerate(models):
        head = model.model[-1]
        hooks.append(head.cpe_projection.register_forward_hook(lambda *_args, i=index: calls.__setitem__(i, calls[i] + 1)))
        model.eval()
    sample = torch.rand(1, 3, 64, 64, device=device)
    with torch.no_grad():
        a, b = models[0](sample), models[1](sample)
    for hook in hooks:
        hook.remove()
    def flatten_tensors(value):
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, dict):
            return [item for key in sorted(value) for item in flatten_tensors(value[key])]
        if isinstance(value, (tuple, list)):
            return [item for child in value for item in flatten_tensors(child)]
        return []

    tensors_a, tensors_b = flatten_tensors(a), flatten_tensors(b)
    inference_identity = bool(tensors_a) and len(tensors_a) == len(tensors_b) and all(
        torch.equal(x, y) for x, y in zip(tensors_a, tensors_b)
    )

    projection = models[0].model[-1].cpe_projection
    projection.train()
    features = [torch.randn(2, layer.in_channels, 2, 2, device=device) for layer in projection.projections]
    labels = torch.tensor([0, 0, 1, 1] * 6, device=device)[: sum(x.shape[2] * x.shape[3] for x in features) * 2]
    embeddings = projection(features).reshape(-1, projection.config.embedding_dim)
    labels = labels[: embeddings.shape[0]]
    base = cpe_supervised_contrastive_loss(embeddings, labels, temperature=0.2)
    grad = {}
    for name, weight in (("control", 0.0), ("candidate", 0.5)):
        projection.zero_grad(set_to_none=True)
        (base * weight).backward(retain_graph=True)
        values = [p.grad for p in projection.parameters()]
        grad[name] = {
            "finite": all(g is not None and bool(torch.isfinite(g).all()) for g in values),
            "sum_abs": sum(float(g.abs().sum()) for g in values if g is not None),
        }
    checks = {
        "configs_differ_only_loss_weight": config_matched and left["cpe"]["loss_weight"] == 0.0 and right["cpe"]["loss_weight"] == 0.5,
        "shared_state_inference_identity": inference_identity,
        "projection_not_called_in_inference": calls == [0, 0],
        "control_projection_gradient_zero": grad["control"]["finite"] and grad["control"]["sum_abs"] == 0.0,
        "candidate_projection_gradient_finite_nonzero": grad["candidate"]["finite"] and grad["candidate"]["sum_abs"] > 0.0,
    }
    result = {
        "format": "coffee_detector.af2_cpe.static_audit.v1",
        "checkpoint_sha256": _sha256(checkpoint),
        "checks": checks,
        "gradient": grad,
        "projection_inference_calls": calls,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "training_authorized": all(checks.values()),
        "test_access_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_static_audit(args.af2_checkpoint, args.output, device=args.device), indent=2))


if __name__ == "__main__":
    main()
