"""E2: fixed Haar frontend on exactly the registered G1/G2 split and schedule."""
import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from ultralytics import YOLO
from coffee_detector.experiments.run_auliya_controlled_generalization import (
    SCHEDULE, MODEL, WEIGHTS_SHA256, ULTRALYTICS_VERSION, prepare, summarize,
    trainer_for, _ensure_runtime, _sha256, _publish_file, _restore_results, _write_json,
)
from coffee_detector.analysis.auliya_gc_error_audit import CHECKPOINTS, RGB_RUNNER_SHA

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
    contract={"protocol":"auliya-controlled-generalization-haar-v1","arm":"HAAR","group":group,"schedule":SCHEDULE,"model":MODEL,
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
            if getattr(model.model,"frequency_arm",None)!="HAAR" or completed!=state["epoch"] or ck.get("optimizer") is None or any(
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
            model.train(trainer=trainer_for("HAAR"),data=str(data_train),save_dir=str(local),
                        resume=str(ll) if completed else False,device=device,plots=False,**SCHEDULE)
        end=json.loads(progress.read_text())
        if end["epoch"]!=80 or _sha256(last)!=end["last_sha256"]:
            raise RuntimeError("Fixed 80 epoch endpoint not reached")
        # No target-group validation during training; evaluate both frozen sets now.
        evaluations={}
        for target in "BC":
            frozen=YOLO(str(last),task="segment")
            if getattr(frozen.model,"frequency_arm",None)!="HAAR":
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
            raise ValueError("HAAR comparison requires the original RGB runtime and split")
    args.output.mkdir(parents=True,exist_ok=True)
    _write_json(args.output/"split_contract.json",split)
    if args.authorize_training:
        train_group(args.group,args.data.resolve(),args.weights,args.output,args.device,split)

if __name__=="__main__":
    main()
