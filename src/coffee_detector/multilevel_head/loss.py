from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from .model import MultilevelResidualDetectHead, _expand_and_clip_boxes


class MultilevelHeadLoss:
    """Native YOLO26 loss plus a capacity-matched ROI classification loss."""

    def __init__(self, model: torch.nn.Module) -> None:
        from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

        self.model = model
        self.head = model.model[-1]
        if not isinstance(self.head, MultilevelResidualDetectHead):
            raise TypeError("MultilevelHeadLoss memerlukan MultilevelResidualDetectHead")
        self.base = (
            E2ELoss(model)
            if getattr(model, "end2end", False)
            else v8DetectionLoss(model)
        )
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

    def _target_rois(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = batch["img"].device
        boxes = batch["bboxes"].to(device)
        labels = batch["cls"].view(-1).long().to(device)
        batch_indices = batch["batch_idx"].view(-1).to(
            device=device, dtype=boxes.dtype
        )
        if not len(boxes):
            empty = boxes.new_zeros((0, 5))
            return empty, labels, empty
        image_height, image_width = map(int, batch["img"].shape[-2:])
        xyxy = torch.cat(
            (boxes[:, :2] - boxes[:, 2:] * 0.5, boxes[:, :2] + boxes[:, 2:] * 0.5),
            dim=1,
        )
        xyxy = xyxy * boxes.new_tensor(
            [image_width, image_height, image_width, image_height]
        )
        matching = torch.cat((batch_indices[:, None], xyxy.detach()), dim=1)
        expanded = _expand_and_clip_boxes(
            xyxy,
            image_height=image_height,
            image_width=image_width,
            factor=self.head.config.box_expand,
        )
        return torch.cat((batch_indices[:, None], expanded), dim=1), labels, matching

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
        matches: list[tuple[int, int]] = []
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

    def _predicted_candidate_loss(
        self,
        parsed: dict[str, dict[str, torch.Tensor]],
        matching_rois: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor | None:
        source_name = self.head.config.candidate_source
        if source_name not in parsed or "one2many" not in parsed:
            return None
        source = parsed[source_name]
        boxes = self.head.base_head._get_decode_boxes(source).transpose(1, 2).detach()
        base_logits = source["scores"].transpose(1, 2).detach()
        features = parsed["one2many"]["feats"]
        boxes = _expand_and_clip_boxes(
            boxes,
            image_height=int(features[0].shape[-2] * float(self.head.stride[0])),
            image_width=int(features[0].shape[-1] * float(self.head.stride[0])),
            factor=self.head.config.box_expand,
        )
        roi_rows = []
        label_rows = []
        logit_rows = []
        for batch_index in range(int(boxes.shape[0])):
            mask = matching_rois[:, 0].long() == batch_index
            targets = matching_rois[mask, 1:]
            labels = target_labels[mask]
            if not len(targets):
                continue
            confidence = base_logits[batch_index].sigmoid().amax(dim=1)
            count = min(int(self.head.config.training_topk), len(confidence))
            indices = confidence.topk(count).indices
            candidates = boxes[batch_index, indices]
            for prediction, target in self._greedy_matches(
                candidates, targets, self.head.config.positive_iou
            ):
                roi_rows.append(
                    torch.cat(
                        (candidates.new_tensor([float(batch_index)]), candidates[prediction])
                    )
                )
                label_rows.append(labels[target])
                logit_rows.append(base_logits[batch_index, indices[prediction]])
        if not roi_rows:
            return None
        residual = self.head.refiner(
            features,
            torch.stack(roi_rows),
            tuple(float(value) for value in self.head.stride),
        )
        logits = torch.stack(logit_rows) + float(
            self.head.config.inference_weight
        ) * residual
        return F.cross_entropy(logits, torch.stack(label_rows))

    def __call__(self, predictions, batch: dict[str, torch.Tensor]):
        base_loss, base_items = self.base(predictions, batch)
        parsed = self._parse(predictions)
        features = parsed["one2many"]["feats"]
        rois, labels, matching = self._target_rois(batch)
        proposal_mix = float(getattr(self.head, "proposal_mix", 0.0))
        zero = sum(parameter.sum() * 0.0 for parameter in self.head.refiner.parameters())
        if len(rois) and proposal_mix < 1.0:
            gt_logits = self.head.refiner(
                features, rois, tuple(float(value) for value in self.head.stride)
            )
            gt_loss = F.cross_entropy(gt_logits, labels)
        else:
            gt_loss = zero
        predicted_loss = (
            self._predicted_candidate_loss(parsed, matching, labels)
            if proposal_mix > 0.0
            else None
        )
        if predicted_loss is None:
            if len(rois) and proposal_mix >= 1.0:
                gt_logits = self.head.refiner(
                    features, rois, tuple(float(value) for value in self.head.stride)
                )
                gt_loss = F.cross_entropy(gt_logits, labels)
            proposal_mix = 0.0
            predicted_loss = gt_loss
        auxiliary = (1.0 - proposal_mix) * gt_loss + proposal_mix * predicted_loss
        weighted = (
            float(self.head.config.auxiliary_weight)
            * auxiliary
            * int(batch["img"].shape[0])
        )
        return (
            torch.cat((base_loss.reshape(-1), weighted.reshape(1))),
            torch.cat((base_items, auxiliary.detach().reshape(1))),
        )
