"""Matched B/C generalization diagnosis; two training runs, four endpoint evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO

from coffee_detector.analysis.auliya_robusta_preflight import (
    METADATA_SHA256, _load_manifest, _polygon_counts, _label_name, _sha256,
)
from coffee_detector.analysis.auliya_representation_audit import HARD
from coffee_detector.experiments.prepare_auliya_official_fold import (
    prepare_official_fold, validate_official_fold, EXTRACTED_CONTENT_SHA256,
)
from coffee_detector.experiments.run_auliya_official_frequency import (
    SCHEDULE as OFFICIAL_SCHEDULE, MODEL, WEIGHTS_SHA256, ULTRALYTICS_VERSION,
    _ensure_runtime, trainer_for,
)
from coffee_detector.experiments.run_auliya_robusta_frequency import (
    _publish_file, _restore_results, _summarize_masks, _write_json,
)

PROTOCOL = "auliya-controlled-generalization-v1"
# Same optimization/augmentation as G0, except fixed endpoint, no early stopping.
SCHEDULE = {**OFFICIAL_SCHEDULE, "patience": 0}
SPLIT_SHA = {
    "B_train":"b6d80a1bd7a803a92fb4ea9da9720f8a39bafd5358dc9be7b544ae42a64f427b",
    "B_eval":"b9cc16811dda68a16e37da20fa824de2693fd9e0ffd3d1c2d0a3c6a59855a81d",
    "C_train":"5255018ecd9ffc84a40ce0cfb3d540c13aaa35e898da1b7d0ab600f22dee8d3c",
    "C_eval":"954ffe78f1b5a28f80dfc0a25d493abd3154eb1b63f8818bf4dc3e76792e5cce",
}
EXPECTED_COUNTS = {"B_train":976,"B_eval":265,"C_train":900,"C_eval":230}


def select(paths, counts, n, tag):
    """Deterministic label-only selection; no learned scores or image pixels."""
    if not 0 < n <= len(paths):
        raise ValueError("Invalid sample size")
    paths = sorted(paths,key=lambda p:hashlib.sha256((tag+"|"+p).encode()).hexdigest())
    x = np.stack([counts[p] for p in paths])
    target = x.sum(0)*n/len(paths)
    scale = np.maximum(target,1)
    cur = np.zeros(21)
    free = np.ones(len(paths),bool)
    chosen = []
    for _ in range(n):
        scores = np.sum(((cur+x-target)/scale)**2,axis=1)
        scores[~free] = np.inf
        i = int(np.argmin(scores))
        free[i] = False
        cur += x[i]
        chosen.append(paths[i])
    return sorted(chosen)


def split_digest(paths):
    return hashlib.sha256(("\n".join(paths)+"\n").encode()).hexdigest()


def plan(manifest, counts):
    partitions = {}
    for g in "BC":
        pool = sorted(p for p in counts if manifest[p]["group"]==g)
        ev = select(pool,counts,75,"GCV1-"+g+"-eval")
        left = sorted(set(pool)-set(ev))
        tr = select(left,counts,300,"GCV1-"+g+"-train") if len(left)>300 else left
        partitions[g+"_train"] = tr
        partitions[g+"_eval"] = ev
    return partitions


def prepare(data, metadata, development):
    if _sha256(metadata) != METADATA_SHA256:
        raise ValueError("Official metadata digest mismatch")
    if not data.exists():
        prepare_official_fold(metadata,development,data)
    validate_official_fold(data)
    original = json.loads((data/"official_fold_contract.json").read_text())
    if original.get("extracted_provenance",{}).get("development_content_sha256") != EXTRACTED_CONTENT_SHA256:
        raise ValueError("Require prior audited extracted dataset")
    with ZipFile(metadata) as z:
        manifest = _load_manifest(z)
    counts = {}
    for p,row in manifest.items():
        if row["group"] not in "BC" or row["group"] not in ("B","C"):
            continue
        label = data/"source"/_label_name(p)
        c = _polygon_counts(label.read_text(encoding="utf-8-sig"),str(label))
        if sum(c.values()) != row["objects"]:
            raise ValueError(f"Manifest instance mismatch: {p}")
        counts[p] = np.array([c[i] for i in range(21)])
    partitions = plan(manifest,counts)
    seen = set()
    summaries = {}
    for key,paths in partitions.items():
        if len(paths)!=(300 if key.endswith("train") else 75) or seen&set(paths):
            raise ValueError("Invalid or overlapping development partition")
        seen.update(paths)
        support = np.stack([counts[p] for p in paths]).sum(0)
        if split_digest(paths)!=SPLIT_SHA[key] or int(support.sum())!=EXPECTED_COUNTS[key] or min(support)<1:
            raise ValueError(f"Frozen split differs: {key}")
        summaries[key] = {"images":len(paths),"instances":int(support.sum()),
                          "class_support":support.tolist(),"split_sha256":SPLIT_SHA[key]}
        target = data/f"gc_{key}.txt"
        value = "".join("./source/"+p+"\n" for p in paths)
        if target.exists() and target.read_text()!=value:
            raise ValueError(f"Existing split differs: {target}")
        target.write_text(value,encoding="utf-8")
    if len(counts)!=795 or len(set(counts)-seen)!=45:
        raise ValueError("Unexpected unused development image inventory")
    for train in "BC":
        for target in "BC":
            path = data/f"gc_{train}_to_{target}.yaml"
            cfg = {"path":str(data.resolve()),"train":f"gc_{train}_train.txt",
                   "val":f"gc_{target}_eval.txt","nc":21,"names":original["names"]}
            if path.exists() and yaml.safe_load(path.read_text())!=cfg:
                raise ValueError("Existing YAML differs")
            path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    report = {"protocol":PROTOCOL,"partitions":summaries,"unused_C_images":45,
              "source_contract_sha256":_sha256(data/"official_fold_contract.json"),
              "source_content_sha256":EXTRACTED_CONTENT_SHA256,
              "test_A_images_accessed":False}
    marker=data/"gc_split_contract.json"
    if marker.exists() and json.loads(marker.read_text())!=report:
        raise ValueError("Existing split contract differs")
    _write_json(marker,report)
    return report


def summarize(evaluated):
    mask = _summarize_masks(evaluated.seg,evaluated.names)
    box = {k.replace("mask","box"):v for k,v in
           _summarize_masks(evaluated.box,evaluated.names).items()}
    result={**mask,**box,"box_map50_21":float(evaluated.box.map50),
            "mask_map50_21":float(evaluated.seg.map50)}
    for kind in ("box","mask"):
        result[f"hard5_{kind}_ap50_95"] = float(np.mean(
            [result[f"{kind}_ap_per_class"][s] for s in HARD]))
    return result


def train_group(group,data,weights,output,device,split):
    _ensure_runtime(weights)
    if group not in ("B","C") or not device.isdecimal() or int(device)>=torch.cuda.device_count():
        raise ValueError("Require group B/C and available single CUDA device")
    root = output/group
    root.mkdir(parents=True,exist_ok=True)
    wdir=root/"weights"
    wdir.mkdir(exist_ok=True)
    last,best=wdir/"last.pt",wdir/"best.pt"
    marker,progress,result=root/"run_contract.json",root/"progress.json",root/"validation_result.json"
    contract={"protocol":PROTOCOL,"group":group,"schedule":SCHEDULE,"model":MODEL,
              "checkpoint_selection":"fixed_last_epoch_80",
              "pretrained_sha256":WEIGHTS_SHA256,"split_contract":split,
              "ultralytics":ULTRALYTICS_VERSION,"torch":str(torch.__version__),
              "gpu":torch.cuda.get_device_name(int(device)),"runner_sha256":_sha256(Path(__file__))}
    if marker.exists():
        if json.loads(marker.read_text())!=contract:
            raise ValueError("Run contract changed")
    elif any(p.exists() for p in (last,best,progress,result)):
        raise ValueError("Unidentified previous run")
    else:
        _write_json(marker,contract)
    if result.exists():
        report=json.loads(result.read_text())
        if report["run_contract"]!=contract or report["checkpoint_sha256"]!=_sha256(last):
            raise ValueError("Stored result mismatch")
        return report
    state=json.loads(progress.read_text()) if progress.exists() else {}
    if bool(state)!=last.exists() or bool(state)!=best.exists():
        raise ValueError("Incomplete checkpoint pair; do not resume mixed epochs")
    with TemporaryDirectory(prefix="auliya-gc-") as tmp:
        local=Path(tmp)/group
        ll,lb=local/"weights/last.pt",local/"weights/best.pt"
        ll.parent.mkdir(parents=True)
        if state:
            for src,dst in ((last,ll),(best,lb)):
                if _sha256(src)!=state[src.stem+"_sha256"]:
                    raise ValueError("Resume checkpoint SHA mismatch")
                shutil.copyfile(src,dst)
            model=YOLO(str(ll),task="segment")
            ck=model.ckpt
            completed=ck["epoch"]+1
            if completed!=state["epoch"] or ck.get("optimizer") is None or any(
                ck["train_args"].get(k)!=v for k,v in SCHEDULE.items()):
                raise ValueError("Resume schedule or epoch mismatch")
            _restore_results(local/"results.csv",ck.get("train_results",{}))
        else:
            completed=0
            model=YOLO(str(weights),task="segment")
        data_train=data/f"gc_{group}_to_{group}.yaml"
        if completed<80:
            def publish(trainer):
                if any(getattr(trainer.args,k)!=v for k,v in SCHEDULE.items()):
                    raise RuntimeError("Training schedule changed")
                _publish_file(trainer.last,last)
                _publish_file(trainer.best,best)
                _publish_file(trainer.csv,root/"results.csv")
                _write_json(progress,{"epoch":trainer.epoch+1,
                    "last_sha256":_sha256(trainer.last),"best_sha256":_sha256(trainer.best)})
            model.add_callback("on_model_save",publish)
            model.train(trainer=trainer_for("RGB"),data=str(data_train),save_dir=str(local),
                        resume=str(ll) if completed else False,device=device,plots=False,**SCHEDULE)
        end=json.loads(progress.read_text())
        if end["epoch"]!=80 or _sha256(last)!=end["last_sha256"]:
            raise RuntimeError("Fixed 80 epoch endpoint not reached")
        # No target-group validation during training; evaluate both frozen sets now.
        evaluations={}
        for target in "BC":
            frozen=YOLO(str(last),task="segment")
            value=frozen.val(data=str(data/f"gc_{group}_to_{target}.yaml"),split="val",
                imgsz=SCHEDULE["imgsz"],batch=SCHEDULE["batch"],workers=SCHEDULE["workers"],
                device=device,project=str(local),name=f"endpoint_{target}",plots=False)
            evaluations[target]=summarize(value)
    report={"run_contract":contract,"completed_epochs":80,
            "checkpoint_sha256":end["last_sha256"],"evaluations":evaluations,
            "test_A_images_accessed":False}
    _write_json(result,report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("development","metadata","data","weights","output"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--group",choices=("B","C"),required=True)
    p.add_argument("--device",default="0")
    p.add_argument("--authorize-training",action="store_true")
    args=p.parse_args()
    _ensure_runtime(args.weights)
    split=prepare(args.data.resolve(),args.metadata,args.development)
    args.output.mkdir(parents=True,exist_ok=True)
    _write_json(args.output/"split_contract.json",split)
    print(json.dumps(split,indent=2),flush=True)
    if args.authorize_training:
        train_group(args.group,args.data.resolve(),args.weights,args.output,args.device,split)


if __name__=="__main__":
    main()
