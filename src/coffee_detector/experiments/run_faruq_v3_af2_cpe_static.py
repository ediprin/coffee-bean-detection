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
ARMS = ("AF2CPE0", "AF2CPE5")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_nc(source: torch.nn.Module) -> int:
    detector = getattr(source, "model", None)
    if isinstance(detector, (torch.nn.Sequential, torch.nn.ModuleList)) and len(detector):
        head = detector[-1]
        if hasattr(head, "nc"):
            return int(head.nc)
    yaml_payload = getattr(source, "yaml", None)
    if isinstance(yaml_payload, dict) and "nc" in yaml_payload:
        return int(yaml_payload["nc"])
    names = getattr(source, "names", None)
    if isinstance(names, (dict, list, tuple)):
        return len(names)
    raise RuntimeError("Tidak dapat menentukan nc dari checkpoint AF2")


def _torch_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    text = str(device).strip()
    if text.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {text} diminta tetapi CUDA tidak tersedia")
        return torch.device(f"cuda:{text}")
    return torch.device(text)


def _state_identical(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    lhs, rhs = left.state_dict(), right.state_dict()
    return lhs.keys() == rhs.keys() and all(torch.equal(lhs[key], rhs[key]) for key in lhs)


def _max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.float() - right.float()).abs().max().detach().cpu())


def _projection_gradient(projection, *, weight: float, device: torch.device) -> dict:
    projection.train()
    projection.zero_grad(set_to_none=True)
    features = [
        torch.randn(2, layer.in_channels, 2, 2, device=device)
        for layer in projection.projections
    ]
    embeddings = projection(features).reshape(-1, projection.config.embedding_dim)
    labels = torch.tensor([0, 0, 1, 1] * 6, device=device, dtype=torch.long)
    if labels.shape[0] != embeddings.shape[0]:
        raise RuntimeError(
            f"Static synthetic labels tidak sejajar: labels={labels.shape[0]} embeddings={embeddings.shape[0]}"
        )
    loss = cpe_supervised_contrastive_loss(
        embeddings, labels, temperature=projection.config.temperature
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("CPE static loss tidak finite")
    (loss * float(weight)).backward()
    gradients = [parameter.grad for parameter in projection.parameters()]
    return {
        "finite": all(g is not None and bool(torch.isfinite(g).all()) for g in gradients),
        "sum_abs": sum(float(g.abs().sum().detach().cpu()) for g in gradients if g is not None),
    }


def run_static_audit(
    checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 128,
) -> dict:
    """Static causal/safety audit following the AF2-FFA precedent.

    Exact identity is proved on the same wrapped/native Detect head with the same
    feature tensors. Separate full CUDA forwards are checked only for numerical
    consistency because AF2 uses FFT kernels and can differ by a few ULPs.
    """
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    torch_device = _torch_device(device)

    configs = {
        arm: yaml.safe_load(
            (ROOT / f"configs/af2_cpe/{arm}_yolo26n.yaml").read_text(encoding="utf-8")
        )
        for arm in ARMS
    }
    control_cfg, candidate_cfg = configs["AF2CPE0"], configs["AF2CPE5"]

    source = YOLO(str(checkpoint)).model.to(torch_device).eval()
    if getattr(source, "afab", None) is None:
        raise RuntimeError("Checkpoint sumber bukan AF2")
    nc = _source_nc(source)

    torch.manual_seed(20260823)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(20260823)
    image = torch.rand(1, 3, image_size, image_size, device=torch_device)
    with torch.inference_mode():
        native = source(image)

    records: dict[str, dict] = {}
    for arm in ARMS:
        payload = configs[arm]
        print(f"[STATIC] build {arm}", flush=True)
        model = AF2CPEDetectionModel(
            str(ROOT / payload["model"]),
            nc=nc,
            verbose=False,
            afab=payload["afab"],
            cpe=payload["cpe"],
        ).to(torch_device)
        transfer = load_fsce_cpe_detector_weights(model, source)
        model.eval()
        head = model.model[-1]

        calls = [0]
        hook = head.cpe_projection.register_forward_hook(
            lambda *_args: calls.__setitem__(0, calls[0] + 1)
        )
        with torch.inference_mode():
            wrapped = model(image)
        hook.remove()

        full_box_diff = _max_abs_difference(
            wrapped[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"]
        )
        full_score_diff = _max_abs_difference(
            wrapped[1]["one2one"]["scores"], native[1]["one2one"]["scores"]
        )

        # Same-head/same-feature proof, copied conceptually from AF2-FFA audit.
        identity_features = [
            torch.rand(1, layer.in_channels, size, size, device=torch_device)
            for layer, size in zip(head.cpe_projection.projections, (16, 8, 4))
        ]
        with torch.inference_mode():
            native_head = head.base_head([feature.clone() for feature in identity_features])
            wrapped_head = head([feature.clone() for feature in identity_features])
        identity_boxes = torch.equal(
            wrapped_head[1]["one2one"]["boxes"], native_head[1]["one2one"]["boxes"]
        )
        identity_scores = torch.equal(
            wrapped_head[1]["one2one"]["scores"], native_head[1]["one2one"]["scores"]
        )

        gradient = _projection_gradient(
            head.cpe_projection,
            weight=float(payload["cpe"]["loss_weight"]),
            device=torch_device,
        )
        source_head = source.model[-1]
        record_gates = {
            "native_head_state_bitwise_preserved": _state_identical(head.base_head, source_head),
            "identity_boxes_bitwise_equal": identity_boxes,
            "identity_scores_bitwise_equal": identity_scores,
            "full_model_numerically_consistent": max(full_box_diff, full_score_diff) <= 1.0e-4,
            "projection_not_called_in_inference": calls == [0],
            "projection_gradient_matches_role": (
                gradient["finite"]
                and (
                    gradient["sum_abs"] == 0.0
                    if arm == "AF2CPE0"
                    else gradient["sum_abs"] > 0.0
                )
            ),
        }
        records[arm] = {
            "loss_weight": float(payload["cpe"]["loss_weight"]),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "state_schema": {key: tuple(value.shape) for key, value in model.state_dict().items()},
            "transfer": transfer,
            "full_model_box_max_abs_diff": full_box_diff,
            "full_model_score_max_abs_diff": full_score_diff,
            "projection_inference_calls": calls[0],
            "gradient": gradient,
            "gates": record_gates,
        }
        print(
            f"[STATIC] {arm}: box_diff={full_box_diff:.3e} score_diff={full_score_diff:.3e} "
            f"grad={gradient['sum_abs']:.6f} gates={record_gates}",
            flush=True,
        )

    left_cpe = {k: v for k, v in control_cfg["cpe"].items() if k != "loss_weight"}
    right_cpe = {k: v for k, v in candidate_cfg["cpe"].items() if k != "loss_weight"}
    global_gates = {
        "same_model_yaml": control_cfg["model"] == candidate_cfg["model"],
        "same_af2_config": control_cfg["afab"] == candidate_cfg["afab"],
        "same_training_schedule": control_cfg["train"] == candidate_cfg["train"],
        "only_loss_weight_differs": left_cpe == right_cpe,
        "control_loss_weight_zero": control_cfg["cpe"]["loss_weight"] == 0.0,
        "candidate_loss_weight_cpe0": candidate_cfg["cpe"]["loss_weight"] == 0.5,
        "same_parameter_count": records["AF2CPE0"]["parameters"] == records["AF2CPE5"]["parameters"],
        "same_state_schema": records["AF2CPE0"]["state_schema"] == records["AF2CPE5"]["state_schema"],
        "test_accessed": False,
    }
    all_arm_gates = all(all(record["gates"].values()) for record in records.values())
    pass_global = all(value for key, value in global_gates.items() if key != "test_accessed")
    decision = "PASS" if all_arm_gates and pass_global and not global_gates["test_accessed"] else "FAIL"

    result = {
        "format": "coffee_detector.af2_cpe.static_audit.v3",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_nc": nc,
        "requested_device": str(device),
        "torch_device": str(torch_device),
        "records": records,
        "gates": global_gates,
        "decision": decision,
        "training_authorized": decision == "PASS",
        "test_access_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[STATIC] decision={decision} output={output}", flush=True)
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
