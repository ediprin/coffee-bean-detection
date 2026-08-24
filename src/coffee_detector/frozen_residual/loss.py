from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from coffee_detector.multilevel_head.model import _expand_and_clip_boxes

from .model import FrozenResidualDetectHead


class FrozenResidualLoss:
    """Train only gated residual classification on matched frozen-D0 boxes."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.head = model.model[-1]
        if not isinstance(self.head, FrozenResidualDetectHead):
            raise TypeError("FrozenResidualLoss memerlukan FrozenResidualDetectHead")
        self.updates = 0
        self.last_matches = 0
        self.last_preserved = 0

    def update(self) -> None:
        self.updates += 1

    @staticmethod
    def _parse(predictions):
        return predictions[1] if isinstance(predictions, tuple) else predictions

    @staticmethod
    def _target_rows(
        batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = batch["img"].device
        boxes = batch["bboxes"].to(device)
        labels = batch["cls"].view(-1).long().to(device)
        batch_indices = batch["batch_idx"].view(-1).to(
            device=device, dtype=boxes.dtype
        )
        if not len(boxes):
            return boxes.new_zeros((0, 5)), labels
        image_height, image_width = map(int, batch["img"].shape[-2:])
        xyxy = torch.cat(
            (boxes[:, :2] - boxes[:, 2:] * 0.5, boxes[:, :2] + boxes[:, 2:] * 0.5),
            dim=1,
        )
        xyxy = xyxy * boxes.new_tensor(
            [image_width, image_height, image_width, image_height]
        )
        return torch.cat((batch_indices[:, None], xyxy), dim=1), labels

    @staticmethod
    def _greedy_matches(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float,
    ) -> list[tuple[int, int]]:
        if not len(predictions) or not len(targets):
            return []
        matrix = box_iou(predictions, targets)
        used_predictions: set[int] = set()
        used_targets: set[int] = set()
        matches = []
        target_count = int(matrix.shape[1])
        for flat_index in matrix.flatten().argsort(descending=True).tolist():
            prediction = int(flat_index // target_count)
            target = int(flat_index % target_count)
            if float(matrix[prediction, target]) < threshold:
                break
            if prediction in used_predictions or target in used_targets:
                continue
            used_predictions.add(prediction)
            used_targets.add(target)
            matches.append((prediction, target))
        return matches

    def _matched_candidates(
        self,
        parsed: dict[str, dict[str, torch.Tensor]],
        targets: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        source = parsed["one2one"]
        features = parsed["one2many"]["feats"]
        boxes = self.head.base_head._get_decode_boxes(source).transpose(1, 2).detach()
        base_logits = source["scores"].transpose(1, 2).detach()
        boxes = _expand_and_clip_boxes(
            boxes,
            image_height=int(features[0].shape[-2] * float(self.head.stride[0])),
            image_width=int(features[0].shape[-1] * float(self.head.stride[0])),
            factor=self.head.frozen_config.box_expand,
        )
        roi_rows = []
        label_rows = []
        logit_rows = []
        for batch_index in range(int(boxes.shape[0])):
            mask = targets[:, 0].long() == batch_index
            batch_targets = targets[mask, 1:]
            batch_labels = labels[mask]
            if not len(batch_targets):
                continue
            confidence = base_logits[batch_index].sigmoid().amax(dim=1)
            count = min(
                int(self.head.frozen_config.training_topk), len(confidence)
            )
            indices = confidence.topk(count).indices
            candidates = boxes[batch_index, indices]
            for prediction, target in self._greedy_matches(
                candidates,
                batch_targets,
                self.head.frozen_config.positive_iou,
            ):
                roi_rows.append(
                    torch.cat(
                        (candidates.new_tensor([float(batch_index)]), candidates[prediction])
                    )
                )
                label_rows.append(batch_labels[target])
                logit_rows.append(base_logits[batch_index, indices[prediction]])
        if not roi_rows:
            return None
        residual = self.head.refiner(
            features,
            torch.stack(roi_rows),
            tuple(float(value) for value in self.head.stride),
        )
        base = torch.stack(logit_rows)
        target_labels = torch.stack(label_rows)
        final, _, correction = self.head.apply_residual(base, residual)
        return final, correction, target_labels

    def __call__(self, predictions, batch: dict[str, torch.Tensor]):
        parsed = self._parse(predictions)
        targets, labels = self._target_rows(batch)
        matched = self._matched_candidates(parsed, targets, labels)
        zero = sum(
            parameter.sum() * 0.0
            for parameter in (
                *self.head.refiner.parameters(),
                *self.head.gate.parameters(),
            )
        )
        if matched is None:
            auxiliary = zero
            self.last_matches = 0
            self.last_preserved = 0
        else:
            final, correction, target_labels = matched
            classification = F.cross_entropy(final, target_labels)
            base_prediction = (final.detach() - correction.detach()).argmax(dim=1)
            correct = base_prediction == target_labels
            preservation = (
                correction[correct].square().mean() if bool(correct.any()) else zero
            )
            auxiliary = classification + float(
                self.head.frozen_config.preservation_weight
            ) * preservation
            self.last_matches = int(len(target_labels))
            self.last_preserved = int(correct.sum())
        batch_size = int(batch["img"].shape[0])
        native_zeros = auxiliary.new_zeros(3)
        loss = torch.cat((native_zeros, (auxiliary * batch_size).reshape(1)))
        items = torch.cat((native_zeros, auxiliary.detach().reshape(1)))
        return loss, items
