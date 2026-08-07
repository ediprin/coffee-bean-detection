from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_local_classifier(num_classes: int, *, imagenet_pretrained: bool = True) -> nn.Module:
    """Lightweight local-stream backbone for the raw-crop diagnostic.

    DC2 evaluates several local-stream extractors; this screening intentionally
    uses one fixed lightweight backbone across all crop resolutions so that the
    controlled factor is raw object resolution, not model capacity.
    """

    weights = MobileNet_V3_Small_Weights.DEFAULT if imagenet_pretrained else None
    model = mobilenet_v3_small(weights=weights)
    if not isinstance(model.classifier[-1], nn.Linear):
        raise TypeError("Unexpected MobileNetV3 classifier schema")
    in_features = int(model.classifier[-1].in_features)
    model.classifier[-1] = nn.Linear(in_features, int(num_classes))
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@torch.inference_mode()
def predict_logits(model: nn.Module, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, labels = [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        logits.append(model(images).cpu())
        labels.append(target.cpu())
    if not logits:
        raise RuntimeError("DataLoader kosong")
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0)
