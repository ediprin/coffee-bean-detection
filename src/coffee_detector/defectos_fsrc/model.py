from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torchvision.ops import box_iou, roi_align

from .operator import stationary_haar_energy


@dataclass(frozen=True)
class FSRCConfig:
    levels: int = 2
    roi_size: int = 4
    hidden_channels: int = 32
    suppression_floor: float = 0.25
    initial_reliability_bias: float = 5.0
    overlap_threshold: float = 0.5

    @classmethod
    def from_mapping(cls, value: Mapping | None = None) -> "FSRCConfig":
        result = value if isinstance(value, cls) else cls(**dict(value or {}))
        if result.levels != 2:
            raise ValueError("FSRC1 dikunci pada dua level stationary Haar")
        if result.roi_size <= 0 or result.hidden_channels <= 0:
            raise ValueError("ROI dan hidden channels harus positif")
        if not 0.0 < result.suppression_floor < 1.0:
            raise ValueError("suppression_floor harus di antara nol dan satu")
        if not 0.0 < result.overlap_threshold < 1.0:
            raise ValueError("overlap_threshold tidak valid")
        return result

    @property
    def spectral_channels(self) -> int:
        return self.levels * 3

    @property
    def feature_dimension(self) -> int:
        # Spectral mean/std/max + score/logit + box geometry + duplicate
        # context + one-hot class identity.
        return self.spectral_channels * 3 + 2 + 4 + 2 + 12


class SpectralReliabilityCalibrator(nn.Module):
    def __init__(self, config: FSRCConfig | Mapping | None = None) -> None:
        super().__init__()
        self.config = FSRCConfig.from_mapping(config)
        self.network = nn.Sequential(
            nn.Linear(self.config.feature_dimension, self.config.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.config.hidden_channels, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, self.config.initial_reliability_bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.config.feature_dimension:
            raise ValueError(
                f"Fitur FSRC harus [N,{self.config.feature_dimension}], "
                f"ditemukan {tuple(features.shape)}"
            )
        return self.network(features).squeeze(1)

    def reliability(self, features: torch.Tensor) -> torch.Tensor:
        return self(features).sigmoid()


def _candidate_context(
    prediction: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    num_classes: int,
    overlap_threshold: float,
) -> torch.Tensor:
    if prediction.ndim != 2 or prediction.shape[1] < 6:
        raise ValueError("Prediction harus [N,6+]")
    if num_classes != 12:
        raise ValueError("FSRC1 dikunci pada 12 kelas DefectosCafeVerde")
    if not len(prediction):
        return prediction.new_zeros((0, 20))
    boxes = prediction[:, :4]
    confidence = prediction[:, 4].clamp(1e-6, 1.0 - 1e-6)
    classes = prediction[:, 5].long()
    width = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0) / float(image_width)
    height = (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0) / float(image_height)
    area = width * height
    log_aspect = (width.clamp_min(1e-6) / height.clamp_min(1e-6)).log().clamp(-4, 4) / 4
    overlap = box_iou(boxes, boxes)
    same_class = classes[:, None] == classes[None, :]
    higher_score = confidence[None, :] > confidence[:, None]
    eligible = same_class & higher_score
    eligible.fill_diagonal_(False)
    masked = overlap.masked_fill(~eligible, 0.0)
    maximum_overlap = masked.max(dim=1).values
    overlap_count = (masked >= overlap_threshold).sum(dim=1).float().clamp_max(10) / 10
    one_hot = torch.nn.functional.one_hot(classes, num_classes=num_classes).to(
        prediction.dtype
    )
    score_logit = torch.logit(confidence).clamp(-8, 8) / 8
    return torch.cat(
        (
            confidence[:, None],
            score_logit[:, None],
            width[:, None],
            height[:, None],
            area[:, None],
            log_aspect[:, None],
            maximum_overlap[:, None],
            overlap_count[:, None],
            one_hot,
        ),
        dim=1,
    )


def candidate_features(
    images: torch.Tensor,
    predictions: list[torch.Tensor],
    config: FSRCConfig | Mapping | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create fixed spectral and candidate-context features.

    Returns the feature matrix and an image-index vector preserving the exact
    order of candidates in ``predictions``.
    """

    cfg = FSRCConfig.from_mapping(config)
    if images.ndim != 4 or len(predictions) != len(images):
        raise ValueError("Batch image dan prediction tidak kompatibel")
    spectral = stationary_haar_energy(images, levels=cfg.levels)
    rois = []
    contexts = []
    image_indices = []
    height, width = images.shape[-2:]
    for index, prediction in enumerate(predictions):
        if not len(prediction):
            continue
        index_column = prediction.new_full((len(prediction), 1), float(index))
        rois.append(torch.cat((index_column, prediction[:, :4]), dim=1))
        contexts.append(
            _candidate_context(
                prediction,
                image_height=height,
                image_width=width,
                num_classes=12,
                overlap_threshold=cfg.overlap_threshold,
            )
        )
        image_indices.append(
            torch.full((len(prediction),), index, device=prediction.device, dtype=torch.long)
        )
    if not rois:
        return (
            images.new_zeros((0, cfg.feature_dimension)),
            torch.empty(0, device=images.device, dtype=torch.long),
        )
    aligned = roi_align(
        spectral,
        torch.cat(rois, dim=0),
        output_size=(cfg.roi_size, cfg.roi_size),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )
    spectral_features = torch.cat(
        (
            aligned.mean(dim=(-2, -1)),
            aligned.std(dim=(-2, -1), unbiased=False),
            aligned.amax(dim=(-2, -1)),
        ),
        dim=1,
    )
    features = torch.cat((spectral_features, torch.cat(contexts, dim=0)), dim=1)
    if features.shape[1] != cfg.feature_dimension or not torch.isfinite(features).all():
        raise RuntimeError("Fitur kandidat FSRC tidak valid")
    return features, torch.cat(image_indices, dim=0)


def apply_reliability_suppression(
    predictions: list[torch.Tensor],
    reliability: torch.Tensor,
    config: FSRCConfig | Mapping | None = None,
    *,
    enabled: bool = True,
) -> list[torch.Tensor]:
    """Adjust only confidence while preserving candidate order, boxes, and classes."""

    cfg = FSRCConfig.from_mapping(config)
    total = sum(len(prediction) for prediction in predictions)
    if reliability.shape != (total,):
        raise ValueError(f"Reliability harus [{total}], ditemukan {tuple(reliability.shape)}")
    if not enabled:
        return [prediction.clone() for prediction in predictions]
    if not torch.isfinite(reliability).all():
        raise ValueError("Reliability tidak finite")
    reliability = reliability.clamp(0.0, 1.0)
    factor = cfg.suppression_floor + (1.0 - cfg.suppression_floor) * reliability
    output = []
    offset = 0
    for prediction in predictions:
        adjusted = prediction.clone()
        count = len(prediction)
        adjusted[:, 4] = prediction[:, 4] * factor[offset : offset + count]
        output.append(adjusted)
        offset += count
    return output
