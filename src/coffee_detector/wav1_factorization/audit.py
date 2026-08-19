from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from coffee_detector.af2_spectral.config import frozen_arm_config as frozen_spectral_arm_config
from coffee_detector.af2_spectral.operator import SpectralInputEnhancer

from .config import ARMS, frozen_arm_config
from .operator import WAV1FactorizationEnhancer


def sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_static_audit(
    d0_checkpoint: str | Path,
    output_path: str | Path,
) -> dict:
    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    torch.manual_seed(20260819)
    image = torch.rand(2, 3, 65, 63, dtype=torch.float32, requires_grad=True)
    old = SpectralInputEnhancer(frozen_spectral_arm_config("WAV1"))
    reference = WAV1FactorizationEnhancer(frozen_arm_config("WAV1_REF"))
    reference_equal = torch.equal(old(image.detach()), reference(image.detach()))

    arms: dict[str, dict] = {}
    all_pass = reference_equal
    for arm in ARMS:
        probe = image.detach().clone().requires_grad_(True)
        frontend = WAV1FactorizationEnhancer(frozen_arm_config(arm))
        first = frontend(probe)
        second = frontend(probe.detach())
        finite = bool(torch.isfinite(first).all())
        shape_ok = tuple(first.shape) == tuple(probe.shape)
        dtype_ok = first.dtype == probe.dtype
        repeatable = torch.equal(first.detach(), second)
        state_free = len(frontend.state_dict()) == 0
        active = not torch.equal(first.detach(), probe.detach())
        first.square().mean().backward()
        finite_grad = probe.grad is not None and bool(torch.isfinite(probe.grad).all())
        passed = finite and shape_ok and dtype_ok and repeatable and state_free and active and finite_grad
        all_pass = all_pass and passed
        arms[arm] = {
            "finite": finite,
            "shape_ok": shape_ok,
            "dtype_ok": dtype_ok,
            "cpu_bitwise_repeatable": repeatable,
            "state_free": state_free,
            "active": active,
            "finite_gradient": finite_grad,
            "decision": "PASS" if passed else "FAIL",
        }

    payload = {
        "format": "coffee_detector.wav1_factorization.static_audit.v1",
        "decision": "PASS" if all_pass else "FAIL",
        "training_authorized": bool(all_pass),
        "wav1_ref_bitwise_equal_to_confirmed_operator": reference_equal,
        "d0_checkpoint_sha256": sha256(checkpoint),
        "test_access_authorized": False,
        "arms": arms,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
