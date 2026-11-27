"""E3 bounded frozen mechanism audit: mask-aware assignment, stability, alternatives."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.tal import SegTaskAlignedAssigner, make_anchors, dist2bbox

from coffee_detector.analysis.auliya_gc_error_audit import (
    CHECKPOINTS, select_validation_head, assign, ranks, box_iou, HARD,
)
from coffee_detector.analysis.auliya_robusta_preflight import _label_name, _sha256
from coffee_detector.experiments import run_auliya_controlled_generalization as gc

VIEWS = ("original","flip","dark","light")
BRANCHES = {"one2many": (10,None), "one2one": (7,1)}
IOU = .75


def scene(data,path):
    bgr=cv2.imread(str(data/"source"/path))
    if bgr is None:
        raise ValueError(f"Image decode failed: {path}")
    h,w=bgr.shape[:2]
    ratio=min(640/h,640/w)
    nw,nh=round(w*ratio),round(h*ratio)
    left,top=round((640-nw)/2-.1),round((640-nh)/2-.1)
    rgb=cv2.cvtColor(cv2.resize(bgr,(nw,nh)),cv2.COLOR_BGR2RGB)
    image=cv2.copyMakeBorder(rgb,top,640-nh-top,left,640-nw-left,
                             cv2.BORDER_CONSTANT,value=(114,114,114))
    y,polys=[],[]
    for line in (data/"source"/_label_name(path)).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip(): continue
        f=line.split()
        y.append(int(f[0]))
        polys.append(np.array(f[1:],np.float32).reshape(-1,2)*[nw,nh]+[left,top])
    gt=np.array([np.r_[p.min(0),p.max(0)] for p in polys],np.float32)
    return image,gt,np.asarray(y,dtype=int),polys,(left,top,nw,nh)


def view_transform(image,gt,region,view):
    x=image.astype(np.float32)/255
    boxes=gt.copy()
    if view=="flip":
        x=x[:,::-1].copy()
        boxes[:,[0,2]]=640-gt[:,[2,0]]
    elif view in ("dark","light"):
        l,t,w,h=region
        # Monotone, no clipping; padding unchanged; no chromatic-channel permutation.
        x[t:t+h,l:l+w]=.9*x[t:t+h,l:l+w]+(.1 if view=="light" else 0.)
    elif view!="original":
        raise ValueError(view)
    return x,boxes


def score_assign(iou,scores,threshold):
    """One-to-one maximum coverage, then confidence; true class not used."""
    n,m=iou.shape
    out=np.full(n,-1,dtype=int)
    if not n or not m: return out
    good=iou>=threshold
    r,c=linear_sum_assignment(good*(n+1+scores[None,:]),maximize=True)
    valid=good[r,c]
    out[r[valid]]=c[valid]
    return out


def decode(model,raw):
    head=model.model[-1]
    anchors,stride=make_anchors(raw["feats"],head.stride,.5)
    boxes=dist2bbox(head.dfl(raw["boxes"]),anchors.T.unsqueeze(0),xywh=False,dim=1).permute(0,2,1)*stride
    scores=raw["scores"].permute(0,2,1).contiguous()
    if boxes.shape!=(1,8400,4) or scores.shape!=(1,8400,21):
        raise ValueError(f"Unexpected predictions: {boxes.shape}, {scores.shape}")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("Nonfinite predictions")
    return boxes,scores,anchors*stride


def infer(model,x,device):
    tensor=torch.from_numpy(x.transpose(2,0,1).copy()).to(device)[None].float()
    (post,_),raw=model(tensor)
    if not all(k in raw for k in BRANCHES):
        raise ValueError("Expected both unfused training branches")
    return post,raw


def assignment_rows(model,raw,gt,y,polys,source,path):
    device=next(model.parameters()).device
    args=model.args
    overlap=bool(args["overlap_mask"])
    ratio=int(args["mask_ratio"])
    if overlap:
        masks,order=polygons2masks_overlap((640,640),polys,downsample_ratio=ratio)
        masks=torch.from_numpy(masks).to(device)[None]
    else:
        order=np.arange(len(y))
        masks=torch.from_numpy(polygons2masks((640,640),polys,1,downsample_ratio=ratio)).to(device)
    ordered_y=torch.tensor(y[order],device=device,dtype=torch.long)[None,:,None]
    ordered_gt=torch.tensor(gt[order],device=device,dtype=torch.float32)[None]
    valid=torch.ones_like(ordered_y,dtype=torch.bool)
    rows=[]
    for branch,(topk,topk2) in BRANCHES.items():
        p=raw[branch]
        boxes,logits,anchors=decode(model,p)
        prob=logits.sigmoid()
        proto=p["proto"][0] if isinstance(p["proto"],tuple) else p["proto"]
        mh,mw=proto.shape[-2:]
        m=masks
        if m.shape[-2:]!=(mh,mw):
            m=torch.nn.functional.interpolate(m[None].float(),(mh,mw),mode="nearest")[0]
        if overlap:
            gm=m[:,None]==torch.arange(1,len(y)+1,device=device)[:,None,None]
        else:
            gm=m[None]>.5
        assigner=SegTaskAlignedAssigner(topk=topk,topk2=topk2,num_classes=21,
            alpha=.5,beta=6.,stride=model.model[-1].stride.tolist()).to(device)
        _,_,target,fg,idx=assigner(prob,boxes,anchors,ordered_y,ordered_gt,valid,
            pd_coef=p["mask_coefficient"].permute(0,2,1),mask_scale=torch.tensor([[mh/640]],device=device),
            proto=proto,gt_masks=gm)
        if not torch.isfinite(target).all():
            raise ValueError("Nonfinite assignment targets")
        # Signed BCE logit derivative, before class/loss weights and normalization.
        grad=prob-target
        iou=box_iou(ordered_gt[0].cpu().numpy(),boxes[0].cpu().numpy())
        for j,orig in enumerate(order):
            sel=fg[0].bool() & (idx[0]==j)
            cls=int(y[orig])
            ts=target[0,sel,cls]
            ps=prob[0,sel,cls]
            gs=grad[0,sel,cls]
            n=int(sel.sum())
            other=grad[0,sel].abs().sum(1)-gs.abs()
            rows.append(dict(source=source,image=path,instance=int(orig),true_class=cls,
                branch=branch,positives=n,nonzero_targets=int((ts>0).sum()),
                target_mass=float(ts.sum()),target_mean=float(ts.mean()) if n else None,
                target_max=float(ts.max()) if n else None,
                predicted_true_mean=float(ps.mean()) if n else None,
                box_iou_mean=float(iou[j,sel.cpu().numpy()].mean()) if n else None,
                true_push_up_mass=float((-gs).clamp(min=0).sum()),
                true_push_down_mass=float(gs.clamp(min=0).sum()),
                competitor_abs_mass=float(other.sum())))
    return rows


def eval_rows(model,post,raw,gt,y,source,target,path,view):
    boxes,logits,_=decode(model,raw["one2many"])
    boxes=np.clip(boxes[0].cpu().numpy(),0,640)
    z=logits[0].cpu().numpy()
    prob=1/(1+np.exp(-np.clip(z,-80,80)))
    iou=box_iou(gt,boxes)
    geom=assign(iou,IOU)
    score=score_assign(iou,prob.max(1),IOU)
    top3=np.argsort(-z,axis=1,kind="stable")[:,:3]
    natives={}
    if view=="original":
        for conf in (.001,.25):
            pred=non_max_suppression(post.clone(),conf_thres=conf,iou_thres=.7,nc=21,
                    multi_label=True,agnostic=False,max_det=300,end2end=False,max_time_img=10.)[0][:,:6].cpu().numpy()
            pred[:,:4]=np.clip(pred[:,:4],0,640)
            niou=box_iou(gt,pred[:,:4])
            # Maximum-cardinality correct-class alternatives, not actual deployed matches.
            correct=assign(np.where(y[:,None]==pred[None,:,5],niou,0),IOU)
            natives[conf]=correct
    rows=[]
    for j,c in enumerate(y):
        row=dict(source=source,target=target,image=path,instance=j,true_class=int(c),view=view)
        for kind,a in (("geometry",geom[j]),("score",score[j])):
            row[kind+"_anchor"]=int(a)
            row[kind+"_iou"]=float(iou[j,a]) if a>=0 else None
            row[kind+"_rank"],row[kind+"_margin"]=ranks(z[a],c) if a>=0 else (None,None)
            row[kind+"_pred"]=int(top3[a,0]) if a>=0 else None
            row[kind+"_true_score"]=float(prob[a,c]) if a>=0 else None
        good=iou[j]>=IOU
        row["any_top1"]=bool(np.any(good & (top3[:,0]==c)))
        row["any_top3"]=bool(np.any(good & np.any(top3==c,axis=1)))
        row["candidate_count"]=int(good.sum())
        row["correct_native_001"]=bool(natives[.001][j]>=0) if natives else None
        row["correct_native_025"]=bool(natives[.25][j]>=0) if natives else None
        rows.append(row)
    return rows


def csv_write(path,rows):
    if not rows: raise ValueError("No records")
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean(values):
    v=[v for v in values if v is not None]
    return float(np.mean(v)) if v else None


def metrics(rows):
    result={"n":len(rows)}
    for k in ("geometry","score"):
        valid=[r for r in rows if r[k+"_anchor"]>=0]
        result[k+"_coverage"]=len(valid)/len(rows)
        result[k+"_top1"]=mean([r[k+"_rank"]==1 for r in valid])
        result[k+"_top3"]=mean([r[k+"_rank"]<=3 for r in valid])
        result[k+"_margin"]=mean([r[k+"_margin"] for r in valid])
    for k in ("any_top1","any_top3","correct_native_001","correct_native_025"):
        result[k]=mean([r[k] for r in rows])
    return result


def summarize(assignments,ev,names):
    hard={names.index(s) for s in HARD}
    groups={"clean18":set(range(18)),"hard5":hard,**{n:{i} for i,n in enumerate(names)}}
    out={"assignment":{},"ranking":{},"stability":{}}
    for s in "BC":
        for branch in BRANCHES:
            for group,ids in groups.items():
                rows=[r for r in assignments if r["source"]==s and r["branch"]==branch and r["true_class"] in ids]
                if not rows: continue
                fields=("positives","nonzero_targets","target_mass","target_mean","target_max","predicted_true_mean",
                        "box_iou_mean","true_push_up_mass","true_push_down_mass","competitor_abs_mass")
                out["assignment"][f"{s}/{branch}/{group}"]={"n":len(rows),"zero_positive":sum(r["positives"]==0 for r in rows),
                    **{k:mean([r[k] for r in rows]) for k in fields}}
        for t in "BC":
            for group,ids in groups.items():
                rows=[r for r in ev if r["source"]==s and r["target"]==t and r["true_class"] in ids]
                if not rows: continue
                base={(r["image"],r["instance"]):r for r in rows if r["view"]=="original"}
                for view in VIEWS:
                    sub=[r for r in rows if r["view"]==view]
                    out["ranking"][f"{s}->{t}/{view}/{group}"]=metrics(sub)
                    if view=="original": continue
                    paired=[(base[(r["image"],r["instance"])],r) for r in sub]
                    for k in ("geometry","score"):
                        both=[(a,b) for a,b in paired if a[k+"_anchor"]>=0 and b[k+"_anchor"]>=0]
                        out["stability"][f"{s}->{t}/{view}/{group}/{k}"]={
                            "n":len(paired),"joint_support":len(both),
                            "class_flip_rate":mean([a[k+"_pred"]!=b[k+"_pred"] for a,b in both]),
                            "base_top1":mean([a[k+"_rank"]==1 for a,b in both]),
                            "view_top1":mean([b[k+"_rank"]==1 for a,b in both]),
                            "rescued":sum(a[k+"_rank"]!=1 and b[k+"_rank"]==1 for a,b in both),
                            "damaged":sum(a[k+"_rank"]==1 and b[k+"_rank"]!=1 for a,b in both),
                            "margin_delta":mean([b[k+"_margin"]-a[k+"_margin"] for a,b in both])}
    return out


def paths(data,part):
    return [p.removeprefix("./source/") for p in (data/f"gc_{part}.txt").read_text().splitlines()]


def run(args):
    if ultralytics.__version__!=gc.ULTRALYTICS_VERSION:
        raise ValueError("Wrong Ultralytics version")
    split=gc.prepare(args.data.resolve(),args.metadata,args.development)
    names=json.loads((args.data/"official_fold_contract.json").read_text())["names"]
    args.output.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    ar,ev=[],[]
    for s in "BC":
        ck=getattr(args,"checkpoint_"+s.lower())
        if _sha256(ck)!=CHECKPOINTS[s]: raise ValueError("Wrong exact checkpoint")
        wrapper=YOLO(str(ck),task="segment")
        if wrapper.ckpt["epoch"]!=79 or any(wrapper.ckpt["train_args"].get(k)!=v for k,v in gc.SCHEDULE.items()):
            raise ValueError("Wrong schedule/epoch")
        model=wrapper.model.to(device).float().eval().requires_grad_(False)
        select_validation_head(model,wrapper.ckpt["train_args"],names)
        with torch.inference_mode():
            for i,path in enumerate(paths(args.data,s+"_train")):
                image,gt,y,polys,region=scene(args.data,path)
                x,_=view_transform(image,gt,region,"original")
                _,raw=infer(model,x,device)
                ar.extend(assignment_rows(model,raw,gt,y,polys,s,path))
                if (i+1)%50==0: print("E3.1",s,i+1,flush=True)
            for t in "BC":
                for i,path in enumerate(paths(args.data,t+"_eval")):
                    image,gt,y,_,region=scene(args.data,path)
                    for view in VIEWS:
                        x,g=view_transform(image,gt,region,view)
                        post,raw=infer(model,x,device)
                        ev.extend(eval_rows(model,post,raw,g,y,s,t,path,view))
                    if (i+1)%25==0: print("E3.2/3",s,t,i+1,flush=True)
        del model,wrapper
        if device.type=="cuda": torch.cuda.empty_cache()
    if len(ar)!=2*(976+900) or len(ev)!=2*len(VIEWS)*(265+230):
        raise ValueError(f"Unexpected record counts: {len(ar)} {len(ev)}")
    csv_write(args.output/"assignment.csv",ar)
    csv_write(args.output/"evaluation.csv",ev)
    gc._write_json(args.output/"report.json",{"protocol":"auliya-gc-e3-v1","split":split,
        "checkpoints":CHECKPOINTS,"runner_sha256":_sha256(Path(__file__)),
        "assignment_records":len(ar),"evaluation_records":len(ev),"views":list(VIEWS),
        "iou_threshold":IOU,"training_executed":False,"test_A_images_accessed":False,
        "results":summarize(ar,ev,names)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("development","metadata","data","checkpoint-b","checkpoint-c","output"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    run(p.parse_args())


if __name__=="__main__":
    main()
