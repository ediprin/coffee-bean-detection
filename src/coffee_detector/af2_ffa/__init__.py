"""AF2 feature-frequency classification adapter for YOLO26."""

from .model import (
    AF2FFAConfig,
    AF2FFADetectHead,
    AF2FFADetectionModel,
    FeatureFrequencyAdapter,
    load_af2_ffa_weights,
)
from .trainer import make_af2_ffa_trainer
from .audit import run_af2_ffa_static_audit
from .from_start_audit import run_af2_ffa_from_start_static_audit
from .dct import DCT_HIGH_FREQUENCY_PAIRS, selected_dct_descriptor

# Backward-compatibility defaults for complete AF2FFAB2 checkpoints serialized
# before the selective-refinement runtime attributes existed. Old nn.Module
# instances are restored without re-running __init__, so class-level fallbacks
# keep ordinary inference behavior identical until an explicit runtime ablation
# override is requested.
FeatureFrequencyAdapter.runtime_strength = 1.0
AF2FFADetectHead.runtime_active_levels = (True, True, True)
AF2FFADetectHead.runtime_fusion_mode = None
AF2FFADetectHead.runtime_residual_mix = None
AF2FFADetectHead.runtime_ambiguity_gate = None
AF2FFADetectHead.runtime_ambiguity_margin = None
AF2FFADetectHead.runtime_ambiguity_temperature = None

__all__ = [
    "AF2FFAConfig",
    "AF2FFADetectHead",
    "AF2FFADetectionModel",
    "FeatureFrequencyAdapter",
    "load_af2_ffa_weights",
    "make_af2_ffa_trainer",
    "run_af2_ffa_static_audit",
    "run_af2_ffa_from_start_static_audit",
    "DCT_HIGH_FREQUENCY_PAIRS",
    "selected_dct_descriptor",
]
