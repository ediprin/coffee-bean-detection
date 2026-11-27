"""E1: frozen G1/G2 box error and rank audit, exact audited checkpoints."""
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
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy

from coffee_detector.analysis.auliya_representation_audit import box_iou, HARD
from coffee_detector.experiments import run_auliya_controlled_generalization as gc
from coffee_detector.analysis.auliya_robusta_preflight import _label_name, _sha256

CHECKPOINTS = {
    "B":"38385effe9335bf5ad23fee9913f19f5373494da1c20bdb578e1138ae52719fc",
    "C":"41ffe7f52503d64ca4a008dd477e0575aa52604c8274f5c7005f1fb0a0723642",
}
RGB_RUNNER_SHA = "2320956abe9399f8920bb0d8e414a0f2cf28806851b01ad0b3fedf2f1c91243a"
THRESHOLDS = (.001,.25)
IOUS = (.5,.75)


def assign(iou, threshold):
    """Maximize number of valid unique matches, then sum IoU. Unmatched=-1."""
    n,m=iou.shape
    out=np.full(n,-1,dtype=int)
    if not n or not m:
        return out
    good=iou>=threshold
    utility=good*(n+1+iou)
    r,c=linear_sum_assignment(utility,maximize=True)
    valid=good[r,c]
    out[r[valid]]=c[valid]
    return out


def ranks(logits,y):
    # Stable class-id tie break, consistent with numpy argmax.
    order=np.argsort(-logits,kind="stable")
    return int(np.flatnonzero(order==y)[0])+1, float(logits[y]-np.max(np.delete(logits,y)))


def load_scene(data,path):
    rgb=cv2.cvtColor(cv2.imread(str(data/"source"/path)),cv2.COLOR_BGR2RGB)
    h,w=rgb.shape[:2]
    r=min(640/h,640/w)
    nw,nh=round(w*r),round(h*r)
    left,top=round((640-nw)/2-.1),round((640-nh)/2-.1)
    rgb=cv2.copyMakeBorder(cv2.resize(rgb,(nw,nh)),top,640-nh-top,left,640-nw-left,
                           cv2.BORDER_CONSTANT,value=(114,114,114))
    y,gt=[],[]
    for line in (data/"source"/_label_name(path)).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip(): continue
        f=line.split()
        p=np.array(f[1:],dtype=np.float32).reshape(-1,2)*[nw,nh]+[left,top]
        gt.append(np.r_[p.min(0),p.max(0)])
        y.append(int(f[0]))
    return rgb,np.asarray(gt),np.asarray(y)


def select_validation_head(model, train_args, names):
    """Match v8.4.210 validator: nms=None/True -> external NMS, False -> O2O."""
    actual = [model.names[i] for i in range(21)]
    if actual != names:
        raise ValueError(f"Class order mismatch: checkpoint={actual!r}; dataset={names!r}")
    if getattr(model.model[-1],"nc",None)!=21:
        raise ValueError("Checkpoint head must have 21 classes")
    if train_args.get("nms") is not None:
        raise ValueError(f"Unexpected historical nms setting: {train_args.get('nms')!r}")
    model.end2end = False
    head=model.model[-1]
    if head.end2end or getattr(head,"cv2",None) is None or getattr(head,"cv3",None) is None:
        raise ValueError("Historical external-NMS head not available")
    if train_args.get("iou") not in (None,.7) or train_args.get("max_det",300)!=300 or train_args.get("agnostic_nms",False):
        raise ValueError("Unexpected historical validation NMS contract")
    print(f"Verified class order; head={type(head).__name__}; end2end={head.end2end}; nms={train_args.get('nms')}; val_iou=0.7",flush=True)


def audit_scene(model,device,data,path,source,target):
    image,gt,y=load_scene(data,path)
    x=torch.from_numpy(image.transpose(2,0,1).copy()).to(device).float()[None]/255
    with torch.inference_mode():
        (post,_),raw=model(x)
        head=model.model[-1]
        # Dual-head checkpoints return both branches; use the same one as validation.
        raw=raw["one2many"] if "one2many" in raw else raw
        decoded=head._get_decode_boxes(raw)[0].T.float()
        if not head.xyxy:
            decoded=xywh2xyxy(decoded)
        boxes=decoded.cpu().numpy()
        logits=raw["scores"][0].T.float().cpu().numpy()
        if post.ndim!=3 or post.shape[0]!=1 or post.shape[2]!=8400:
            raise ValueError(f"Expected external-NMS BCN output, got {tuple(post.shape)}")
        # Validation NMS uses multi_label=True, class-aware, IoU=.7, max_det=300.
        # Run at each threshold separately, as the validator would.
        native={}
        for conf in THRESHOLDS:
            pred=non_max_suppression(post.clone(),conf_thres=conf,iou_thres=.7,
                nc=21,multi_label=True,agnostic=False,max_det=300,end2end=False,
                max_time_img=10.0)[0][:,:6].float().cpu().numpy()
            if pred.ndim!=2 or pred.shape[1]!=6 or not np.isfinite(pred).all():
                raise ValueError("Unexpected NMS output")
            pred[:,:4]=np.clip(pred[:,:4],0,640)
            native[conf]=pred
    if boxes.shape!=(8400,4) or logits.shape!=(8400,21):
        raise ValueError("Unexpected raw output")
    if not all(np.isfinite(v).all() for v in (boxes,logits)):
        raise ValueError("Nonfinite outputs")
    boxes=np.clip(boxes,0,640)
    iou=box_iou(gt,boxes)
    rows=[]
    for threshold in IOUS:
        raw_match=assign(iou,threshold)
        # Raw rank is conditional on a geometry-only unique anchor assignment.
        for conf in THRESHOLDS:
            pred=native[conf]
            piou=box_iou(gt,pred[:,:4])
            match=assign(piou,threshold)
            for k,c in enumerate(y):
                j,a=match[k],raw_match[k]
                row=dict(source=source,target=target,image=path,instance=k,true_class=int(c),
                         iou_threshold=threshold,confidence=conf,raw_access=bool(a>=0),
                         raw_iou=float(iou[k,a]) if a>=0 else None,
                         raw_rank=None,raw_margin=None,raw_top1=None,raw_true_score=None,
                         native_access=bool(j>=0),native_iou=float(piou[k,j]) if j>=0 else None,
                         native_class=int(pred[j,5]) if j>=0 else None,
                         native_score=float(pred[j,4]) if j>=0 else None)
                if a>=0:
                    rank,margin=ranks(logits[a],c)
                    row.update(raw_rank=rank,raw_margin=margin,raw_top1=int(np.argmax(logits[a])),
                               raw_true_score=float(1/(1+np.exp(-np.clip(logits[a,c],-80,80)))))
                row["native_outcome"]=("correct" if j>=0 and int(pred[j,5])==c else
                    "wrong_class" if j>=0 else "miss")
                rows.append(row)
    return rows


def summarize(rows,names):
    result={}
    hard={names.index(s) for s in HARD}
    for iou in IOUS:
        for conf in THRESHOLDS:
            subset=[r for r in rows if r["iou_threshold"]==iou and r["confidence"]==conf]
            for key,group in [("clean18",[r for r in subset if r["true_class"]<18]),
                              ("hard5",[r for r in subset if r["true_class"] in hard]),
                              *[(names[c],[r for r in subset if r["true_class"]==c]) for c in range(21)]]:
                if not group: continue
                raw=[r for r in group if r["raw_access"]]
                native=[r for r in group if r["native_access"]]
                outcomes={k:sum(r["native_outcome"]==k for r in group) for k in ("correct","wrong_class","miss")}
                result[f"iou{iou}_conf{conf}_{key}"]={
                    "n":len(group),"raw_access":len(raw),"native_access":len(native),
                    "raw_top1_conditional":sum(r["raw_rank"]==1 for r in raw)/len(raw) if raw else None,
                    "raw_top3_conditional":sum(r["raw_rank"]<=3 for r in raw)/len(raw) if raw else None,
                    "raw_margin_median":float(np.median([r["raw_margin"] for r in raw])) if raw else None,
                    "native_conditional_accuracy":outcomes["correct"]/len(native) if native else None,
                    "native_outcomes":outcomes}
    return result


def run(args):
    if ultralytics.__version__!=gc.ULTRALYTICS_VERSION:
        raise ValueError("Ultralytics version mismatch")
    split=gc.prepare(args.data.resolve(),args.metadata,args.development)
    names=json.loads((args.data/"official_fold_contract.json").read_text())["names"]
    args.output.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    results={}
    all_rows=[]
    for source in "BC":
        ck=getattr(args,"checkpoint_"+source.lower())
        if _sha256(ck)!=CHECKPOINTS[source]:
            raise ValueError(f"Exact G1/G2 epoch-80 checkpoint required: {source}")
        wrapper=YOLO(str(ck),task="segment")
        if wrapper.ckpt["epoch"]!=79 or any(wrapper.ckpt["train_args"].get(k)!=v for k,v in gc.SCHEDULE.items()):
            raise ValueError("Checkpoint epoch/schedule mismatch")
        model=wrapper.model.to(device).float().eval().requires_grad_(False)
        select_validation_head(model,wrapper.ckpt["train_args"],names)
        for target in "BC":
            paths=[p.removeprefix("./source/") for p in (args.data/f"gc_{target}_eval.txt").read_text().splitlines()]
            rows=[]
            for path in paths:
                rows.extend(audit_scene(model,device,args.data,path,source,target))
            if len(rows)!=gc.EXPECTED_COUNTS[target+"_eval"]*len(IOUS)*len(THRESHOLDS):
                raise ValueError("Unexpected audit instance count")
            results[source+"_to_"+target]=summarize(rows,names)
            all_rows.extend(rows)
            print(source,target,results[source+"_to_"+target]["iou0.5_conf0.25_clean18"],flush=True)
        del model,wrapper
        if device.type=="cuda": torch.cuda.empty_cache()
    with (args.output/"instance_diagnostics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    gc._write_json(args.output/"report.json",{"protocol":"auliya-gc-e1-v2","split":split,
        "checkpoints":CHECKPOINTS,"runner_sha256":_sha256(Path(__file__)),
        "inference":"square640_FP32_one2many_external_class_aware_NMS_iou0.7_multi_label_maxdet300; geometry-only bipartite matching",
        "results":results,"test_A_images_accessed":False,"training_executed":False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("development","metadata","data","checkpoint-b","checkpoint-c","output"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    run(p.parse_args())


if __name__=="__main__":
    main()
