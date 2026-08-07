from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, roi_align

from coffee_detector.coffee_fg.model import _expand_and_clip_boxes
from .model import MRLConfig, MRLDetectHead


def square_ring_masks(size: int, *, device=None) -> tuple[torch.Tensor, ...]:
    """Return center-to-outer square-ring masks used by SFRNet Fig. 6.

    For the paper's default 7x7 RoI this yields four groups: the center pixel,
    the surrounding 3x3 ring, the 5x5 ring, and the 7x7 outer ring.
    """

    if size <= 0 or size % 2 == 0:
        raise ValueError("Grouped Euclidean distance memerlukan ukuran RoI ganjil")
    coordinates = torch.arange(size, device=device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    center = size // 2
    radius = torch.maximum((yy - center).abs(), (xx - center).abs())
    return tuple(radius.eq(index) for index in range(center + 1))


def grouped_euclidean_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """SFRNet Eq. (3): mean L2 distance between center-to-outer group means."""

    if left.shape != right.shape or left.ndim != 4:
        raise ValueError("Grouped distance membutuhkan dua tensor [N,C,H,W] berukuran sama")
    if left.shape[-1] != left.shape[-2]:
        raise ValueError("MRL memakai RoI persegi")
    masks = square_ring_masks(int(left.shape[-1]), device=left.device)
    distances = []
    for mask in masks:
        positions = mask.reshape(-1)
        left_mean = left.flatten(2)[:, :, positions].mean(dim=2)
        right_mean = right.flatten(2)[:, :, positions].mean(dim=2)
        distances.append(torch.linalg.vector_norm(left_mean - right_mean, ord=2, dim=1))
    return torch.stack(distances, dim=1).mean(dim=1)


def multi_roi_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """SFRNet Eq. (4): mean softplus(d_G(anchor,pos)-d_G(anchor,neg))."""

    if not len(anchor):
        return anchor.sum() * 0.0
    d_positive = grouped_euclidean_distance(anchor, positive)
    d_negative = grouped_euclidean_distance(anchor, negative)
    return F.softplus(d_positive - d_negative).mean()


class MRLDetectionLoss:
    """Native YOLO26 loss plus SFRNet-style proposal-level Multi-RoI regularization."""

    def __init__(self, model: torch.nn.Module) -> None:
        from ultralytics.utils.loss import E2ELoss

        self.model = model
        self.head = model.model[-1]
        if not isinstance(self.head, MRLDetectHead):
            raise TypeError("MRLDetectionLoss memerlukan MRLDetectHead")
        self.config = MRLConfig.from_mapping(self.head.config)
        self.base = E2ELoss(model)
        self.updates = int(getattr(self.base, "updates", 0))

    def update(self) -> None:
        update = getattr(self.base, "update", None)
        if update is None:
            return
        self.base.updates = int(self.updates)
        update()
        self.updates = int(self.base.updates)

    @staticmethod
    def _parse(predictions):
        return predictions[1] if isinstance(predictions, tuple) else predictions

    @staticmethod
    def _target_boxes(batch: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        images = batch["img"]
        device = images.device
        height, width = (int(value) for value in images.shape[-2:])
        normalized = batch["bboxes"].to(device)
        labels = batch["cls"].view(-1).long().to(device)
        batch_indices = batch["batch_idx"].view(-1).long().to(device)
        xyxy = torch.cat(
            (normalized[:, :2] - normalized[:, 2:] * 0.5, normalized[:, :2] + normalized[:, 2:] * 0.5),
            dim=1,
        )
        xyxy = xyxy * normalized.new_tensor([width, height, width, height])
        boxes_by_image, labels_by_image = [], []
        for image_index in range(images.shape[0]):
            mask = batch_indices.eq(image_index)
            boxes_by_image.append(xyxy[mask])
            labels_by_image.append(labels[mask])
        return boxes_by_image, labels_by_image

    def _collect_proposal_rois(
        self,
        parsed: dict[str, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Build S = T foreground proposals + up to T random background proposals."""

        if "one2many" not in parsed:
            return None
        source = parsed["one2many"]
        features = source["feats"]
        boxes = self.head.base_head._get_decode_boxes(source).transpose(1, 2).detach()
        scores = source["scores"].transpose(1, 2).detach()
        image_height = int(features[0].shape[-2] * float(self.head.stride[0]))
        image_width = int(features[0].shape[-1] * float(self.head.stride[0]))
        target_boxes, target_labels = self._target_boxes(batch)

        foreground_rows: list[torch.Tensor] = []
        foreground_labels: list[torch.Tensor] = []
        background_rows: list[torch.Tensor] = []
        for image_index in range(boxes.shape[0]):
            confidence = scores[image_index].sigmoid().amax(dim=1)
            count = min(int(self.config.training_topk), int(confidence.numel()))
            if count <= 0:
                continue
            indices = confidence.topk(count).indices
            candidates = boxes[image_index, indices]
            candidates = _expand_and_clip_boxes(
                candidates,
                image_height=image_height,
                image_width=image_width,
                factor=float(self.config.box_expand),
            )
            if len(target_boxes[image_index]):
                overlaps = box_iou(candidates, target_boxes[image_index])
                max_iou, assignment = overlaps.max(dim=1)
                is_foreground = max_iou.ge(float(self.config.foreground_iou))
                for candidate, target_index in zip(candidates[is_foreground], assignment[is_foreground]):
                    foreground_rows.append(
                        torch.cat((candidate.new_tensor([float(image_index)]), candidate))
                    )
                    foreground_labels.append(target_labels[image_index][target_index])
                for candidate in candidates[~is_foreground]:
                    background_rows.append(
                        torch.cat((candidate.new_tensor([float(image_index)]), candidate))
                    )
            else:
                for candidate in candidates:
                    background_rows.append(
                        torch.cat((candidate.new_tensor([float(image_index)]), candidate))
                    )

        if len(foreground_rows) < 2:
            return None
        foreground = torch.stack(foreground_rows)
        labels = torch.stack(foreground_labels).long()
        if background_rows:
            background = torch.stack(background_rows)
            take = min(len(background), len(foreground))
            selection = torch.randperm(len(background), device=background.device)[:take]
            background = background[selection]
        else:
            background = foreground.new_zeros((0, 5))

        all_rois = torch.cat((foreground, background), dim=0)
        background_labels = labels.new_full((len(background),), int(self.head.nc))
        all_labels = torch.cat((labels, background_labels), dim=0)
        return all_rois, all_labels

    def _triad_indices(self, labels: torch.Tensor, foreground_count: int) -> tuple[torch.Tensor, ...] | None:
        """Traverse every eligible foreground anchor once, sampling paper-style triads."""

        anchors, positives, negatives = [], [], []
        foreground_labels = labels[:foreground_count]
        for anchor in range(foreground_count):
            same = torch.nonzero(foreground_labels.eq(foreground_labels[anchor]), as_tuple=False).flatten()
            same = same[same.ne(anchor)]
            if not len(same):
                continue
            different = torch.nonzero(labels.ne(foreground_labels[anchor]), as_tuple=False).flatten()
            if not len(different):
                continue
            pos = same[torch.randint(len(same), (1,), device=labels.device)].item()
            neg = different[torch.randint(len(different), (1,), device=labels.device)].item()
            anchors.append(anchor)
            positives.append(int(pos))
            negatives.append(int(neg))
        if not anchors:
            return None
        return (
            labels.new_tensor(anchors, dtype=torch.long),
            labels.new_tensor(positives, dtype=torch.long),
            labels.new_tensor(negatives, dtype=torch.long),
        )

    def _mrl(self, parsed: dict[str, dict[str, torch.Tensor]], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        source = parsed.get("one2many")
        if source is None:
            return batch["img"].sum() * 0.0
        collected = self._collect_proposal_rois(parsed, batch)
        if collected is None:
            return sum(feature.sum() * 0.0 for feature in source["feats"])
        rois, labels = collected
        foreground_count = int(labels.ne(int(self.head.nc)).sum())
        triads = self._triad_indices(labels, foreground_count)
        if triads is None:
            return sum(feature.sum() * 0.0 for feature in source["feats"])

        level = int(self.config.feature_level)
        feature = source["feats"][level]
        aligned = roi_align(
            feature,
            rois.to(device=feature.device, dtype=feature.dtype),
            output_size=(int(self.config.roi_size), int(self.config.roi_size)),
            spatial_scale=1.0 / float(self.head.stride[level]),
            sampling_ratio=2,
            aligned=True,
        )
        anchor_idx, positive_idx, negative_idx = triads
        return multi_roi_loss(aligned[anchor_idx], aligned[positive_idx], aligned[negative_idx])

    def __call__(self, predictions, batch: dict[str, torch.Tensor]):
        base_loss, base_items = self.base(predictions, batch)
        parsed = self._parse(predictions)
        metric = self._mrl(parsed, batch)
        batch_size = int(batch["img"].shape[0])
        weighted = float(self.config.loss_weight) * metric * batch_size
        total = torch.cat((base_loss.reshape(-1), weighted.reshape(1)))
        items = torch.cat((base_items, metric.detach().reshape(1)))
        return total, items
