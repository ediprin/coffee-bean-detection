"""D1/D2 augmentation-only comparison on frozen G1/G2 split."""
import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from ultralytics import YOLO
from coffee_detector.experiments.run_auliya_controlled_generalization import (
    SCHEDULE, MODEL, WEIGHTS_SHA256, ULTRALYTICS_VERSION, prepare, summarize,
    _ensure_runtime, _sha256, _publish_file, _restore_results, _write_json,
)
from coffee_detector.analysis.auliya_gc_error_audit import CHECKPOINTS, RGB_RUNNER_SHA
from coffee_detector.auliya_diversification import ARMS, P, trainer_for, check_train_pipeline, oa_photo
from coffee_detector import auliya_diversification as aug
import numpy as np
import cv2
from coffee_detector.analysis import auliya_gc_mechanism_audit as e3

PROTOCOL="auliya-gc-diversification-v1"

def schedule_for(arm):
    return {**SCHEDULE,"copy_paste":P if arm=="D1" else 0.0,
            "copy_paste_mode":"mixup" if arm=="D1" else "flip"}

def previews(data,output,group):
    path=output/"previews"/group
    path.mkdir(parents=True,exist_ok=True)
    report=[]
    for i,p in enumerate(e3.paths(data,group+"_train")[:8]):
        image,gt,y,_,_=e3.scene(data,p)
        bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        changed=oa_photo(bgr,gt,np.random.RandomState(1000+i))
        if changed.shape!=bgr.shape or changed.dtype!=bgr.dtype:
            raise RuntimeError("Invalid photometric augmentation")
        cv2.imwrite(str(path/f"{i:02d}.jpg"),np.concatenate((bgr,changed),axis=1))
        report.append({"image":p,"mean_absolute_pixel_delta":float(np.abs(bgr.astype(float)-changed).mean())})
    if not any(r["mean_absolute_pixel_delta"]>0 for r in report):
        raise RuntimeError("No active OA-Mix adaptation")
    _write_json(path/"preview.json",report)


def train_group(arm,group,data,weights,output,device,split):
    schedule=schedule_for(arm)
    _ensure_runtime(weights)
    if group not in ("B","C") or not device.isdecimal() or int(device)>=torch.cuda.device_count():
        raise ValueError("Require group B/C and available single CUDA device")
    root = output/arm/group
    root.mkdir(parents=True,exist_ok=True)
    wdir=root/"weights"
    wdir.mkdir(exist_ok=True)
    last,best=wdir/"last.pt",wdir/"best.pt"
    marker,progress,result=root/"run_contract.json",root/"progress.json",root/"validation_result.json"
    contract={"protocol":PROTOCOL,"arm":arm,"group":group,"schedule":schedule,
              "augmentation_sha256":_sha256(Path(aug.__file__)),
              "d2_probability":P if arm=="D2" else 0.,"model":MODEL,
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
                ck["train_args"].get(k)!=v for k,v in schedule.items()):
                raise ValueError("Resume schedule or epoch mismatch")
            _restore_results(local/"results.csv",ck.get("train_results",{}))
        else:
            completed=0
            model=YOLO(str(weights),task="segment")
        data_train=data/f"gc_{group}_to_{group}.yaml"
        if completed<80:
            def publish(trainer):
                if any(getattr(trainer.args,k)!=v for k,v in schedule.items()):
                    raise RuntimeError("Training schedule changed")
                _publish_file(trainer.last,last)
                _publish_file(trainer.best,best)
                _publish_file(trainer.csv,root/"results.csv")
                _write_json(progress,{"epoch":trainer.epoch+1,
                    "last_sha256":_sha256(trainer.last),"best_sha256":_sha256(trainer.best)})
            model.add_callback("on_model_save",publish)
            # Native close_mosaic turns off both extra policies in final 10 epochs.
            if completed < 70:
                model.add_callback("on_train_start",lambda trainer:check_train_pipeline(trainer,arm))
            model.train(trainer=trainer_for(arm),data=str(data_train),save_dir=str(local),
                        resume=str(ll) if completed else False,device=device,plots=False,**schedule)
        end=json.loads(progress.read_text())
        if end["epoch"]!=80 or _sha256(last)!=end["last_sha256"]:
            raise RuntimeError("Fixed 80 epoch endpoint not reached")
        # No target-group validation during training; evaluate both frozen sets now.
        evaluations={}
        for target in "BC":
            frozen=YOLO(str(last),task="segment")
            if getattr(frozen.model,"frequency_arm","RGB")!="RGB":
                raise ValueError("Wrong trained frontend")
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
    for k in ("development","metadata","data","weights","output","rgb-results"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--group",choices=("B","C"),required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--device",default="0")
    p.add_argument("--authorize-training",action="store_true")
    args=p.parse_args()
    _ensure_runtime(args.weights)
    split=prepare(args.data.resolve(),args.metadata,args.development)
    # Reject a different runtime or historical RGB reference.
    for group in "BC":
        rgb=json.loads((args.rgb_results/group/"validation_result.json").read_text())
        c=rgb["run_contract"]
        if (rgb["checkpoint_sha256"]!=CHECKPOINTS[group] or rgb["completed_epochs"]!=80
            or rgb["test_A_images_accessed"] is not False or c["split_contract"]!=split
            or c["runner_sha256"]!=RGB_RUNNER_SHA or c["schedule"]!=SCHEDULE
            or c["pretrained_sha256"]!=WEIGHTS_SHA256 or c["ultralytics"]!=ULTRALYTICS_VERSION
            or c["torch"]!=str(torch.__version__) or c["gpu"]!=torch.cuda.get_device_name(int(args.device))):
            raise ValueError("D1/D2 comparison requires the original RGB runtime and split")
    args.output.mkdir(parents=True,exist_ok=True)
    _write_json(args.output/"split_contract.json",split)
    if args.arm=="D2":
        previews(args.data,args.output,args.group)
    if args.authorize_training:
        train_group(args.arm,args.group,args.data.resolve(),args.weights,args.output,args.device,split)

if __name__=="__main__":
    main()
