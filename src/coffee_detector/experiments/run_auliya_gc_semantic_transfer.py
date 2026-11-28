"""S1/S2 semantic transfer screening, frozen G1/G2 protocol."""
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
from coffee_detector.auliya_semantic_transfer import ARMS, trainer_for
from coffee_detector import auliya_semantic_transfer as method
from coffee_detector.experiments.run_auliya_gc_competitive import rank_audit
from coffee_detector.experiments.run_auliya_official_frequency import trainer_for as native_trainer
from types import SimpleNamespace
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import E2ELoss, SegmentLoss26
from copy import deepcopy

PROTOCOL="auliya-gc-semantic-transfer-v1"

def preflight(weights,data,device):
    original=YOLO(str(weights),task="segment").model
    names=json.loads((data/"official_fold_contract.json").read_text())["names"]
    models={}
    for arm in ("S0",*ARMS):
        cls=native_trainer("RGB") if arm=="S0" else trainer_for(arm)
        t=object.__new__(cls)
        t.data={"nc":21,"channels":3,"names":dict(enumerate(names))}
        t.args=SimpleNamespace(cls_remap=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SCHEDULE["seed"])
            models[arm]=t.get_model(original.yaml,original,verbose=False).eval()
    native=models["S0"].state_dict()
    for arm in ARMS:
        state=models[arm].state_dict()
        if any(k not in state or not torch.equal(state[k],v) for k,v in native.items()):
            raise RuntimeError("Native initialization changed: "+arm)
    image=torch.rand((2,3,128,128),generator=torch.Generator().manual_seed(123)).to(device)
    masks=torch.zeros((2,32,32),device=device,dtype=torch.uint8)
    masks[:,8:24,8:24]=1
    sem=torch.zeros((2,32,32),device=device,dtype=torch.long)
    sem[0,8:24,8:24]=1
    sem[1,8:24,8:24]=7
    batch={"img":image,"batch_idx":torch.tensor([0.,1.],device=device),
           "cls":torch.tensor([[1.],[7.]],device=device),
           "bboxes":torch.tensor([[.5,.5,.5,.5],[.5,.5,.5,.5]],device=device),
           "masks":masks,"sem_masks":sem}
    stats={}
    reference=None
    for arm,m in models.items():
        m=m.to(device).train()
        m.args=get_cfg(overrides={"task":"segment","mode":"train","imgsz":128})
        with torch.no_grad():
            raw=m(image)
            loss,_=E2ELoss(m,SegmentLoss26)(raw,batch)
        if not torch.isfinite(loss).all():
            raise RuntimeError("Nonfinite native loss")
        tensors=[raw[b][k] for b in ("one2many","one2one")
                 for k in ("boxes","scores","mask_coefficient")]
        tensors.extend(raw["one2many"]["proto"])
        tensors.append(loss)
        tensors=[v.detach().cpu() for v in tensors]
        if reference is None: reference=tensors
        elif not all(torch.allclose(a,b,rtol=1e-5,atol=1e-5) for a,b in zip(tensors,reference)):
            raise RuntimeError("Initial native output/loss changed: "+arm)
        counts={}
        for mode in (False,True):
            fused=deepcopy(m).eval()
            fused.end2end=mode
            fused.fuse(verbose=False)
            with torch.no_grad():
                out=fused(image)
            if not torch.isfinite(out[0][0]).all():
                raise RuntimeError("Fused output invalid")
            counts[str(mode)]=sum(p.numel() for p in fused.parameters())
        stats[arm]={"train_parameters":sum(p.numel() for p in m.parameters()),
                    "fused_parameters":counts,"initial_loss":loss.cpu().tolist()}
        m.cpu()
    if stats["S1"]["fused_parameters"]!=stats["S2"]["fused_parameters"]:
        raise RuntimeError("Inference capacity differs")
    return {"native_initial_state_and_loss_equal":True,"stats":stats,
            "method_sha256":_sha256(Path(method.__file__)),
            "runner_sha256":_sha256(Path(__file__)),
            "pretrained_sha256":WEIGHTS_SHA256,
            "training_executed":False}


def train_group(arm,group,data,weights,output,device,split):
    _ensure_runtime(weights)
    if group not in ("B","C") or not device.isdecimal() or int(device)>=torch.cuda.device_count():
        raise ValueError("Require group B/C and available single CUDA device")
    root = output/arm/group
    root.mkdir(parents=True,exist_ok=True)
    wdir=root/"weights"
    wdir.mkdir(exist_ok=True)
    last,best=wdir/"last.pt",wdir/"best.pt"
    marker,progress,result=root/"run_contract.json",root/"progress.json",root/"validation_result.json"
    contract={"protocol":PROTOCOL,"arm":arm,"group":group,"schedule":SCHEDULE,"model":MODEL,
              "method_sha256":_sha256(Path(method.__file__)),
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
            if getattr(model.model,"semantic_arm",None)!=arm or completed!=state["epoch"] or ck.get("optimizer") is None or any(
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
            model.train(trainer=trainer_for(arm),data=str(data_train),save_dir=str(local),
                        resume=str(ll) if completed else False,device=device,plots=False,**SCHEDULE)
        end=json.loads(progress.read_text())
        if end["epoch"]!=80 or _sha256(last)!=end["last_sha256"]:
            raise RuntimeError("Fixed 80 epoch endpoint not reached")
        # No target-group validation during training; evaluate both frozen sets now.
        evaluations={}
        for target in "BC":
            frozen=YOLO(str(last),task="segment")
            if getattr(frozen.model,"semantic_arm",None)!=arm:
                raise ValueError("Wrong trained semantic arm")
            value=frozen.val(data=str(data/f"gc_{group}_to_{target}.yaml"),split="val",
                imgsz=SCHEDULE["imgsz"],batch=SCHEDULE["batch"],workers=SCHEDULE["workers"],
                device=device,project=str(local),name=f"endpoint_{target}",plots=False)
            evaluations[target]=summarize(value)
    ranking=rank_audit(last,data,group,torch.device("cuda:"+device),root)
    report={"run_contract":contract,"completed_epochs":80,"ranking":ranking,
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
    p.add_argument("--preflight-only",action="store_true")
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
            raise ValueError("Semantic transfer comparison requires the original RGB runtime and split")
    args.output.mkdir(parents=True,exist_ok=True)
    _write_json(args.output/"split_contract.json",split)
    if args.preflight_only:
        check=preflight(args.weights,args.data,torch.device("cuda:"+args.device))
        _write_json(args.output/"preflight.json",check)
        print(json.dumps(check,indent=2),flush=True)
    elif args.authorize_training:
        check=json.loads((args.output/"preflight.json").read_text())
        if (check.get("native_initial_state_and_loss_equal") is not True
            or check.get("method_sha256")!=_sha256(Path(method.__file__))
            or check.get("runner_sha256")!=_sha256(Path(__file__))
            or check.get("pretrained_sha256")!=WEIGHTS_SHA256):
            raise ValueError("Invalid or stale preflight")
        train_group(args.arm,args.group,args.data.resolve(),args.weights,args.output,args.device,split)

if __name__=="__main__":
    main()
