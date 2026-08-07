from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from .model import CoffeeFGDetectHead, _expand_and_clip_boxes


class CoffeeFGLoss:
    """YOLO26 detection loss plus an isolated object-level class loss."""

    def __init__(self, model: torch.nn.Module) -> None:
        from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

        self.model = model
        self.head = model.model[-1]
        if not isinstance(self.head, CoffeeFGDetectHead):
            raise TypeError("CoffeeFGLoss memerlukan CoffeeFGDetectHead")
        self.base = E2ELoss(model) if getattr(model, "end2end", False) else v8DetectionLoss(model)
        self.updates = int(getattr(self.base, "updates", 0))

    def update(self) -> None:
        """Forward YOLO26's resumed end-to-end loss schedule to the base loss."""

        update = getattr(self.base, "update", None)
        if update is None:
            return
        # BaseTrainer restores ``criterion.updates`` on the outer wrapper.
        # Keep E2ELoss on that same schedule before advancing it once.
        self.base.updates = int(self.updates)
        update()
        self.updates = int(self.base.updates)

    @staticmethod
    def _parse(predictions):
        return predictions[1] if isinstance(predictions, tuple) else predictions

    def _target_rois(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Ultralytics keeps ``head.stride`` as a CPU metadata tensor even when
        # the model and images have moved to CUDA. Targets must follow the
        # actual compute tensor, otherwise the auxiliary CE receives CUDA
        # logits and CPU labels.
        compute_device = batch["img"].device
        boxes = batch["bboxes"].to(compute_device)
        labels = batch["cls"].view(-1).long().to(boxes.device)
        batch_indices = batch["batch_idx"].view(-1).to(device=boxes.device, dtype=boxes.dtype)
        if not len(boxes):
            empty = boxes.new_zeros((0, 5))
            return empty, labels, empty

        image_height, image_width = (int(value) for value in batch["img"].shape[-2:])
        centres = boxes[:, :2]
        sizes = boxes[:, 2:]
        xyxy = torch.cat((centres - sizes * 0.5, centres + sizes * 0.5), dim=1)
        xyxy = xyxy * boxes.new_tensor(
            [image_width, image_height, image_width, image_height]
        )
        matching_rois = torch.cat((batch_indices[:, None], xyxy.detach().clone()), dim=1)
        expanded_xyxy = _expand_and_clip_boxes(
            xyxy,
            image_height=image_height,
            image_width=image_width,
            factor=self.head.config.box_expand,
        )
        return (
            torch.cat((batch_indices[:, None], expanded_xyxy), dim=1),
            labels,
            matching_rois,
        )

    @staticmethod
    def _greedy_matches(
        predicted_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        threshold: float,
    ) -> list[tuple[int, int]]:
        if not len(predicted_boxes) or not len(target_boxes):
            return []
        matrix = box_iou(predicted_boxes, target_boxes)
        order = matrix.flatten().argsort(descending=True)
        used_predictions: set[int] = set()
        used_targets: set[int] = set()
        matches = []
        target_count = matrix.shape[1]
        for flat_index in order.tolist():
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
        target_matching_rois: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor | None:
        source_name = self.head.config.candidate_source
        if source_name not in parsed or "one2many" not in parsed:
            return None
        source = parsed[source_name]
        boxes = self.head.base_head._get_decode_boxes(source).transpose(1, 2).detach()
        base_logits = source["scores"].transpose(1, 2).detach()
        original_features = parsed["one2many"]["feats"]
        boxes = _expand_and_clip_boxes(
            boxes,
            image_height=int(
                original_features[0].shape[-2] * float(self.head.stride[0])
            ),
            image_width=int(
                original_features[0].shape[-1] * float(self.head.stride[0])
            ),
            factor=self.head.config.box_expand,
        )
        roi_rows = []
        label_rows = []
        logit_rows = []
        for batch_index in range(boxes.shape[0]):
            mask = target_matching_rois[:, 0].long() == batch_index
            batch_targets = target_matching_rois[mask, 1:]
            batch_labels = target_labels[mask]
            if not len(batch_targets):
                continue
            confidence = base_logits[batch_index].sigmoid().amax(dim=1)
            count = min(self.head.config.training_topk, len(confidence))
            indices = confidence.topk(count).indices
            candidates = boxes[batch_index, indices]
            matches = self._greedy_matches(
                candidates,
                batch_targets,
                self.head.config.positive_iou,
            )
            for prediction, target in matches:
                roi_rows.append(
                    torch.cat(
                        (
                            candidates.new_tensor([float(batch_index)]),
                            candidates[prediction],
                        )
                    )
                )
                label_rows.append(batch_labels[target])
                logit_rows.append(base_logits[batch_index, indices[prediction]])
        if not roi_rows:
            return None
        rois = torch.stack(roi_rows)
        labels = torch.stack(label_rows)
        fixed_base_logits = torch.stack(logit_rows)
        residual = self.head.refiner(
            original_features,
            rois,
            tuple(float(value) for value in self.head.stride),
        )
        final_logits = (
            fixed_base_logits
            + float(self.head.config.inference_weight) * residual
        )
        return F.cross_entropy(final_logits, labels)

    def __call__(self, predictions, batch: dict[str, torch.Tensor]):
        base_loss, base_items = self.base(predictions, batch)
        parsed = self._parse(predictions)
        branch = parsed["one2many"] if "one2many" in parsed else parsed
        features = branch["feats"]
        rois, labels, matching_rois = self._target_rois(batch)
        proposal_mix = float(getattr(self.head, "proposal_mix", 0.0))
        zero = sum(
            parameter.sum() * 0.0 for parameter in self.head.refiner.parameters()
        )
        if len(rois) and proposal_mix < 1.0:
            logits = self.head.refiner(
                features,
                rois,
                tuple(float(value) for value in self.head.stride),
            )
            gt_auxiliary = F.cross_entropy(logits, labels)
        else:
            gt_auxiliary = zero
        predicted_auxiliary = (
            self._predicted_candidate_loss(parsed, matching_rois, labels)
            if proposal_mix > 0.0
            else None
        )
        if predicted_auxiliary is None:
            if len(rois) and proposal_mix >= 1.0:
                logits = self.head.refiner(
                    features,
                    rois,
                    tuple(float(value) for value in self.head.stride),
                )
                gt_auxiliary = F.cross_entropy(logits, labels)
            proposal_mix = 0.0
            predicted_auxiliary = gt_auxiliary
        auxiliary = (
            (1.0 - proposal_mix) * gt_auxiliary
            + proposal_mix * predicted_auxiliary
        )

        batch_size = int(batch["img"].shape[0])
        auxiliary_weighted = (
            float(self.head.config.auxiliary_weight) * auxiliary * batch_size
        )
        total = torch.cat((base_loss.reshape(-1), auxiliary_weighted.reshape(1)))
        items = torch.cat((base_items, auxiliary.detach().reshape(1)))
        return total, items
