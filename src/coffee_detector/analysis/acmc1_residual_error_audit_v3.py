"""ACMC1 residual-error audit with legacy-checkpoint compatibility.

Older ACMC1 checkpoints predate the ACMC2 ``entropy_margin_gates`` attribute.
PyTorch restores pickled module state without re-running ``__init__``, so those
valid entropy-only checkpoints can lack the newer attribute when loaded by the
current class definition. This entrypoint installs a narrow compatibility shim
before loading any checkpoint, then delegates to the validation-only V2 audit.
"""

from __future__ import annotations

import torch

from coffee_detector.ambiguity_multilevel.model import AmbiguityConditionedFusion
from coffee_detector.analysis import acmc1_residual_error_audit_v2 as audit_v2


def _legacy_compatible_ambiguity(
    self: AmbiguityConditionedFusion,
    logits: torch.Tensor,
    level: int,
) -> torch.Tensor:
    """Treat a missing ACMC2 gate attribute as the original ACMC1 entropy gate."""
    probability = logits.detach().softmax(dim=1)
    entropy = self._entropy(probability)
    gates = getattr(self, "entropy_margin_gates", None)
    if gates is None:
        return entropy
    margin_uncertainty = self._margin_uncertainty(probability)
    return gates[level](entropy, margin_uncertainty)


def install_legacy_acmc1_checkpoint_compatibility() -> None:
    AmbiguityConditionedFusion._ambiguity = _legacy_compatible_ambiguity


def main() -> None:
    install_legacy_acmc1_checkpoint_compatibility()
    audit_v2.main()


if __name__ == "__main__":
    main()
