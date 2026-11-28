"""P3 semantic-reference sharing, with an inference-capacity-matched control.

S1: independent copy of native semantic trunk, trained by detection cls only.
S2: shared native semantic trunk, trained by native semantic + detection cls.
Both retain native semantic loss, masks, box regression and assignment.
Only P3 logits receive a zero-initialized residual. No new objective.
"""
from copy import deepcopy

import torch
from torch import nn
from ultralytics.nn.modules.block import Proto
from ultralytics.nn.modules.head import Segment26, Detect
from coffee_detector.experiments.run_auliya_official_frequency import trainer_for as native_trainer

ARMS = ("S1", "S2")


class SemanticTransferHead(Segment26):
    """Native Segment26 output contract; explicit shared reference computation."""

    def _reference_and_proto(self, x):
        # Exact Proto26 fusion, once, not a second pass through BatchNorm.
        u = x[0]
        for i, conv in enumerate(self.proto.feat_refine):
            up = torch.nn.functional.interpolate(conv(x[i+1]), scale_factor=2**(i+1), mode="nearest")
            u = u + up
        p = Proto.forward(self.proto, self.proto.feat_fuse(u))
        sem = self.proto.semseg
        if self.semantic_arm == "S2":
            ref = sem[1](sem[0](u))
            if self.training:
                p = (p, sem[2](ref))
        else:
            ref = self.independent_reference(u)
            if self.training:
                p = (p, sem(u))
        return ref, p

    def _correct(self, x, branch, reference, residual):
        raw = self.forward_head(x, **branch)
        if not raw:
            return raw
        delta = residual(reference).flatten(2)
        n = x[0].shape[-2] * x[0].shape[-1]
        # Preserve P4/P5 logits; avoid in-place modification of native output.
        z = raw["scores"]
        raw["scores"] = torch.cat((z[..., :n] + delta, z[..., n:]), dim=-1)
        return raw

    def forward(self, x):
        ref, proto = self._reference_and_proto(x)
        many = self._correct(x, self.one2many, ref, self.o2m_residual) if self.cv2 is not None else {}
        if getattr(self, "one2one_cv2", None) is not None:
            # Preserve native one-to-one isolation from backbone and reference.
            xs = [v.detach() for v in x] if self.training else x
            r = ref.detach() if self.training else ref
            one = self._correct(xs, self.one2one, r, self.o2o_residual)
            many["proto"] = proto
            one["proto"] = tuple(v.detach() for v in proto) if isinstance(proto, tuple) else proto.detach()
            raw = {"one2many": many, "one2one": one}
            selected = one if self.end2end else many
        else:
            many["proto"] = proto
            raw = selected = many
        if self.training:
            return raw
        y = self._inference(selected)
        if self.end2end:
            y = self.postprocess(y.permute(0,2,1))
        return (y, proto) if self.export else ((y, proto), raw)

    def fuse(self):
        # Match native branch removal under both external-NMS and O2O modes.
        Detect.fuse(self)
        if self.end2end:
            self.o2m_residual = None
        else:
            self.o2o_residual = None
        if self.semantic_arm == "S2":
            # Retain the two reference convolutions, not the semantic classifier.
            self.proto.semseg = nn.Sequential(*list(self.proto.semseg.children())[:2])
        else:
            self.proto.fuse()


def install(model, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    h = model.model[-1]
    if type(h) is not Segment26 or not h.end2end:
        raise TypeError("Require unfused native dual-head Segment26")
    sem = h.proto.semseg
    if len(sem) != 3 or not isinstance(sem[-1], nn.Conv2d):
        raise RuntimeError("Unexpected native semantic branch")
    h.__class__ = SemanticTransferHead
    h.semantic_arm = model.semantic_arm = arm
    # Same initial values in both reference trunks, including BN buffers.
    h.independent_reference = deepcopy(sem[:2]) if arm == "S1" else None
    for name in ("o2m_residual", "o2o_residual"):
        r = nn.Conv2d(sem[-1].in_channels, h.nc, 1).to(sem[-1].weight)
        nn.init.zeros_(r.weight)
        nn.init.zeros_(r.bias)
        setattr(h, name, r)
    h.train(h.training)
    return model


class SemanticControlTrainer(native_trainer("RGB")):
    arm = "S1"

    def get_model(self, cfg=None, weights=None, verbose=True):
        source_arm = getattr(weights, "semantic_arm", None)
        if source_arm is not None and source_arm != self.arm:
            raise ValueError("Cannot resume another semantic-transfer arm")
        model = super().get_model(cfg, weights, verbose)
        install(model, self.arm)
        if source_arm is not None:
            # Native loader runs first to remap the original cls tower width.
            # Restore ALL tensors strictly, including the newly installed paths.
            model.load_state_dict(weights.float().state_dict(), strict=True)
        return model


class SemanticSharedTrainer(SemanticControlTrainer):
    arm = "S2"


def trainer_for(arm):
    return {"S1": SemanticControlTrainer, "S2": SemanticSharedTrainer}[arm]
