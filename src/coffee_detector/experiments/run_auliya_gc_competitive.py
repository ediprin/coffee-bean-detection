"""M1/M2 paired competitive supervision on the frozen G1/G2 split."""
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
from coffee_detector.analysis.auliya_gc_error_audit import CHECKPOINTS, RGB_RUNNER_SHA, select_validation_head, HARD
from coffee_detector.auliya_competitive import trainer_for, ARMS, MARGIN, LAMBDA
from coffee_detector.analysis import auliya_gc_mechanism_audit as e3
from coffee_detector import auliya_competitive
from types import SimpleNamespace
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import E2ELoss, SegmentLoss26
import numpy as np

PROTOCOL="auliya-gc-competitive-v1"


def preflight(weights,data,device):
    # Build exactly the native model and both loss variants with identical seed.
    from coffee_detector.experiments.run_auliya_official_frequency import trainer_for as native_trainer
    original=YOLO(str(weights),task="segment").model
    names=json.loads((data/"official_fold_contract.json").read_text())["names"]
    models={}
    for arm in ("M0",*ARMS):
        cls=native_trainer("RGB") if arm=="M0" else trainer_for(arm)
        t=object.__new__(cls)
        t.data={"nc":21,"channels":3,"names":dict(enumerate(names))}
        t.args=SimpleNamespace(cls_remap=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SCHEDULE["seed"])
            models[arm]=t.get_model(original.yaml,original,verbose=False).eval()
    native=models["M0"].state_dict()
    for arm in ARMS:
        state=models[arm].state_dict()
        if list(state)!=list(native) or any(not torch.equal(state[k],v) for k,v in native.items()):
            raise RuntimeError(f"Initial parameters differ: {arm}")
    # Use an identical synthetic batch to exercise both actual native mask-aware
    # assignment branches, auxiliary gradients and loss-vector integration.
    model=models["M0"].to(device).train()
    model.args=get_cfg(overrides={"task":"segment","mode":"train","imgsz":128})
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
    pred=model(image)
    native_criterion=E2ELoss(model,SegmentLoss26)
    base,items=native_criterion(pred,batch)
    checks={}
    for arm in ARMS:
        criterion=auliya_competitive.CompetitiveCriterion(model,arm)
        loss,logged=criterion(pred,batch)
        aux=criterion.last_aux
        if criterion.one2many.last_positive<=0 or not torch.isfinite(aux) or float(aux)<=0:
            raise RuntimeError("Auxiliary inactive on synthetic batch")
        if not torch.allclose(loss[[0,1,3,4]],base[[0,1,3,4]],rtol=1e-5,atol=1e-5):
            raise RuntimeError("Native nonclassification losses changed")
        expected=aux*LAMBDA*len(image)
        if not torch.allclose(loss[2]-base[2],expected,rtol=1e-4,atol=1e-5):
            raise RuntimeError("Native cls integration mismatch")
        if any(not torch.allclose(logged[k],items[k]) for k in items):
            raise RuntimeError("Native logged components changed")
        grad=torch.autograd.grad(loss.sum()-base.sum(),pred["one2many"]["scores"],retain_graph=True)[0]
        if not torch.isfinite(grad).all() or not bool(grad.abs().sum()>0):
            raise RuntimeError("Auxiliary classification gradient missing")
        checks[arm]={"aux":float(aux),"positive_anchors":criterion.one2many.last_positive,
                     "gradient_abs_sum":float(grad.abs().sum())}
    return {"identical_initial_state":True,"parameter_count":sum(p.numel() for p in model.parameters()),
            "synthetic_checks":checks,"training_executed":False}


def rank_audit(checkpoint,data,source,device):
    wrapper=YOLO(str(checkpoint),task="segment")
    names=json.loads((data/"official_fold_contract.json").read_text())["names"]
    model=wrapper.model.to(device).float().eval().requires_grad_(False)
    select_validation_head(model,wrapper.ckpt["train_args"],names)
    hard={names.index(s) for s in HARD}
    results={}
    with torch.inference_mode():
        for t in "BC":
            rows=[]
            for path in e3.paths(data,t+"_eval"):
                image,gt,y,_,region=e3.scene(data,path)
                x,g=e3.view_transform(image,gt,region,"original")
                post,raw=e3.infer(model,x,device)
                rows.extend(e3.eval_rows(model,post,raw,g,y,source,t,path,"original"))
            if len(rows)!=e3.gc.EXPECTED_COUNTS[t+"_eval"]:
                raise ValueError("Ranking support mismatch")
            results[t]={"clean18":e3.metrics([r for r in rows if r["true_class"]<18]),
                        "hard5":e3.metrics([r for r in rows if r["true_class"] in hard])}
    return results


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
    contract={"protocol":PROTOCOL,"arm":arm,"margin":MARGIN,"lambda":LAMBDA,
              "auxiliary_scope":"O2M fixed-weight outside native branch decay",
              "loss_sha256":_sha256(Path(auliya_competitive.__file__)),"group":group,"schedule":SCHEDULE,"model":MODEL,
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
            if getattr(model.model,"competitive_arm",None)!=arm or completed!=state["epoch"] or ck.get("optimizer") is None or any(
                ck["train_args"].get(k)!=v for k,v in SCHEDULE.items()):
                raise ValueError("Resume schedule or epoch mismatch")
            _restore_results(local/"results.csv",ck.get("train_results",{}))
        else:
            completed=0
            model=YOLO(str(weights),task="segment")
        data_train=data/f"gc_{group}_to_{group}.yaml"
        if completed<80:
            aux_stats=[]
            def track(trainer):
                crit=trainer.model.criterion
                if crit.arm!=arm or crit.last_aux is None:
                    raise RuntimeError("Wrong or inactive training criterion")
                aux_stats.append(float(crit.last_aux))
            def log_aux(trainer):
                if not aux_stats or not np.isfinite(aux_stats).all():
                    raise RuntimeError("No finite auxiliary batch log")
                with (root/"auxiliary.jsonl").open("a") as f:
                    f.write(json.dumps({"epoch":trainer.epoch+1,"batches":len(aux_stats),"mean_aux":float(np.mean(aux_stats)),
                                       "native_o2m_weight":trainer.model.criterion.o2m})+"\n")
                aux_stats.clear()
            model.add_callback("on_train_batch_end",track)
            model.add_callback("on_train_epoch_end",log_aux)
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
            if getattr(frozen.model,"competitive_arm",None)!=arm:
                raise ValueError("Wrong trained competitive arm")
            value=frozen.val(data=str(data/f"gc_{group}_to_{target}.yaml"),split="val",
                imgsz=SCHEDULE["imgsz"],batch=SCHEDULE["batch"],workers=SCHEDULE["workers"],
                device=device,project=str(local),name=f"endpoint_{target}",plots=False)
            evaluations[target]=summarize(value)
    ranking=rank_audit(last,data,group,torch.device("cuda:"+device))
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
            raise ValueError("M1/M2 comparison requires the original RGB runtime and split")
    args.output.mkdir(parents=True,exist_ok=True)
    _write_json(args.output/"split_contract.json",split)
    if args.preflight_only:
        result=preflight(args.weights,args.data,torch.device("cuda:"+args.device))
        _write_json(args.output/"preflight.json",result)
        print(json.dumps(result,indent=2),flush=True)
    elif args.authorize_training:
        if not (args.output/"preflight.json").is_file():
            raise ValueError("Run real-model preflight first")
        train_group(args.arm,args.group,args.data.resolve(),args.weights,args.output,args.device,split)

if __name__=="__main__":
    main()
