from .calibration import PatchScaleReport, calibrate_patch_size, choose_patch_size, collect_equivalent_box_sides
from .config import CAFRConfig, VARIANTS, frozen_variant_config
from .model import CAFRDetectionModel, load_cafr_weights, make_cafr_trainer
from .operator import CAFRInputEnhancer, rgb_luminance, shared_residual_gate, soft_spectral_weight

__all__ = [
    "CAFRConfig",
    "VARIANTS",
    "frozen_variant_config",
    "CAFRInputEnhancer",
    "rgb_luminance",
    "shared_residual_gate",
    "soft_spectral_weight",
    "CAFRDetectionModel",
    "load_cafr_weights",
    "make_cafr_trainer",
    "PatchScaleReport",
    "collect_equivalent_box_sides",
    "choose_patch_size",
    "calibrate_patch_size",
]
