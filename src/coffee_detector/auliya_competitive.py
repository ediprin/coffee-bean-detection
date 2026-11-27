"""Training-only paired competitive supervision for Auliya G1/G2."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import E2ELoss, SegmentLoss26
from ultralytics.nn.modules.head import Segment26
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import RANK

ARMS=("M1","M2")
MARGIN=1.0
LAMBDA=0.25


def competitive_loss(logits, labels, group, arm, margin=MARGIN):
    """Positive anchors only; one unit of weight per represented GT.

    M1: strongest competitor across anchors vs true logit at that anchor.
    M2: same strongest competitor vs maximum true logit across those anchors.
    Stable first-anchor tie break, same hard selection in both arms.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    z=logits.float()
    if z.shape[0]==0:
        return z.sum()*0.
    if z.ndim!=2 or labels.shape!=(len(z),) or group.shape!=(len(z),):
        raise ValueError("Invalid competitive loss input shape")
    _, inv=torch.unique(group,sorted=True,return_inverse=True)
    n=int(inv.max())+1
    pos=z.gather(1,labels[:,None]).squeeze(1)
    neg=z.masked_fill(F.one_hot(labels,num_classes=z.shape[1]).bool(),-torch.inf).max(1).values
    # Selection is discrete; gather selected logits to retain ordinary autograd.
    maxneg=z.new_full((n,),-torch.inf).scatter_reduce_(0,inv,neg.detach(),reduce="amax",include_self=True)
    index=torch.arange(len(z),device=z.device)
    neg_idx=torch.full((n,),len(z),dtype=torch.long,device=z.device).scatter_reduce_(
        0,inv,torch.where(neg.detach()==maxneg[inv],index,len(z)),reduce="amin",include_self=True)
    if arm=="M1":
        pos_idx=neg_idx
    else:
        maxpos=z.new_full((n,),-torch.inf).scatter_reduce_(0,inv,pos.detach(),reduce="amax",include_self=True)
        pos_idx=torch.full((n,),len(z),dtype=torch.long,device=z.device).scatter_reduce_(
            0,inv,torch.where(pos.detach()==maxpos[inv],index,len(z)),reduce="amin",include_self=True)
    return F.softplus(float(margin)+neg[neg_idx]-pos[pos_idx]).mean()


def positive_logits(preds,batch,fg,idx):
    """Map native padded per-image GT indices to flat batch target labels."""
    z=preds["scores"].permute(0,2,1)
    bi=batch["batch_idx"].view(-1).long().to(z.device)
    labels=batch["cls"].view(-1).long().to(z.device)
    if len(bi)!=len(labels):
        raise ValueError("Target label count mismatch")
    b,a=fg.nonzero(as_tuple=True)
    if len(b)==0:
        return z[fg],labels[:0],bi[:0]
    # Native preprocess assumes image-grouped targets. Verify rather than remap silently.
    if not bool((bi[1:]>=bi[:-1]).all()):
        raise ValueError("Expected image-grouped native target rows")
    counts=torch.bincount(bi,minlength=z.shape[0])
    offsets=counts.cumsum(0)-counts
    local=idx[b,a].long()
    if bool(((local<0)|(local>=counts[b])).any()):
        raise ValueError("Native target index out of bounds")
    g=offsets[b]+local
    return z[b,a],labels[g],g


class CompetitiveSegmentLoss(SegmentLoss26):
    """Native mask-aware assignment and native loss; capture one auxiliary scalar."""
    def __init__(self,model,arm):
        super().__init__(model,tal_topk=10)
        self.arm=arm
        self.aux=None
        self.last_positive=0

    def get_assigned_targets_and_loss(self,preds,batch):
        assigned,loss,items=super().get_assigned_targets_and_loss(preds,batch)
        fg,idx,*_=assigned
        z,y,g=positive_logits(preds,batch,fg,idx)
        self.aux=competitive_loss(z,y,g,self.arm)
        self.last_positive=int(z.shape[0])
        return assigned,loss,items


class CompetitiveCriterion(E2ELoss):
    """Keep the original O2M/O2O decay and append fixed-weight auxiliary to cls slot."""
    def __init__(self,model,arm):
        super().__init__(model,SegmentLoss26)
        self.one2many=CompetitiveSegmentLoss(model,arm)
        self.arm=arm
        self.last_aux=None

    def __call__(self,preds,batch):
        # Call the unmodified native dual-head objective, including its logging.
        loss,items=super().__call__(preds,batch)
        aux=self.one2many.aux
        self.one2many.aux=None  # do not retain the autograd graph
        if aux is None:
            raise RuntimeError("Auxiliary loss was not executed")
        self.last_aux=aux.detach()
        # Native returns 5 loss components, each multiplied by batch size.
        # Keep native loss items (O2O) unchanged for comparability; log aux separately.
        if loss.shape!=(5,):
            raise RuntimeError(f"Unexpected loss vector {loss.shape}")
        scale=preds["one2many"]["scores"].shape[0] if isinstance(preds,dict) else preds[1]["one2many"]["scores"].shape[0]
        extra=torch.stack((aux*0,aux*0,aux*LAMBDA*scale,aux*0,aux*0))
        return loss+extra,items


class CompetitiveSegmentationModel(SegmentationModel):
    """Identical native architecture; only the training criterion differs."""
    def __init__(self,cfg,*,arm,ch=3,nc=None,verbose=True):
        super().__init__(cfg,ch=ch,nc=nc,verbose=verbose)
        if arm not in ARMS:
            raise ValueError(arm)
        self.competitive_arm=arm

    def init_criterion(self):
        if not isinstance(self.model[-1],Segment26) or getattr(self.model[-1],"one2one_cv2",None) is None:
            raise ValueError("Require dual-head YOLO26 segmentation")
        return CompetitiveCriterion(self,self.competitive_arm)


class CompetitiveTrainer(SegmentationTrainer):
    arm=None

    def get_model(self,cfg=None,weights=None,verbose=True):
        model=self.set_model_names_for_load(CompetitiveSegmentationModel(
            cfg,arm=self.arm,nc=self.data["nc"],ch=self.data["channels"],verbose=verbose and RANK==-1))
        if weights:
            model.load(weights)
        return model

    def final_eval(self):
        # Same fixed-epoch protocol as G1/G2; keep checkpoint optimizer.
        pass


class M1Trainer(CompetitiveTrainer):
    arm="M1"


class M2Trainer(CompetitiveTrainer):
    arm="M2"


def trainer_for(arm):
    return {"M1":M1Trainer,"M2":M2Trainer}[arm]
