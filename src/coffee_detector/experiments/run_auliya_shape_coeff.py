"""Frozen B-to-C pilot: fresh RGB or SHAPECOEF1; defaults to synthetic preflight only."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.metrics import box_iou

from coffee_detector.analysis.auliya_box_consensus import matched_indices
from coffee_detector.analysis.auliya_robusta_preflight import _sha256
from coffee_detector.auliya_frequency_model import FrequencySegmentationModel
from coffee_detector.auliya_shape_coeff import ARM, FOCUS, ShapeCoeffModel, shape_regression, targets_from_masks
from coffee_detector.experiments.run_auliya_robusta_frequency import (
    BATCH, EPOCHS, IMAGE_SIZE, SEED, ULTRALYTICS_VERSION, WEIGHTS_SHA256, _development_yaml,
    _load_stage, _publish_file, _restore_results, _summarize_masks, _write_json, _trainer_for,
)


PROTOCOL = 'auliya-bc-shape-coeff-v1'
DEVELOPMENT_SHA = '23d653ca2e98c185a76bb7b2095d6afc6401876cbe48fd219cb42bdc7ecbd360'


class ShapeCoeffTrainer(SegmentationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = ShapeCoeffModel(cfg, nc=self.data['nc'], ch=self.data['channels'], verbose=verbose)
        model = self.set_model_names_for_load(model)
        if weights is not None:
            model.load(weights)
        return model

    def preprocess_batch(self, batch):
        batch['shape_target'], batch['shape_valid'] = targets_from_masks(batch, overlap=self.args.overlap_mask)
        return super().preprocess_batch(batch)

    def final_eval(self):
        # The experiment reports the fixed final epoch and retains optimizer state for resume.
        pass


def verify(development, weights):
    if ultralytics.__version__ != ULTRALYTICS_VERSION or _sha256(weights) != WEIGHTS_SHA256:
        raise ValueError('Use Ultralytics 8.4.96 and the pinned official YOLO26n-seg weights')
    data = _development_yaml(development.resolve())
    if _sha256(data.parent / 'development_contract.json') != DEVELOPMENT_SHA:
        raise ValueError('Unexpected B/C development contract')
    return data


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def preflight(development, weights, device='cpu'):
    data = verify(development, weights)
    names = json.loads((data.parent / 'development_contract.json').read_text(encoding='utf-8'))['names']
    device = torch.device(f'cuda:{device}' if device.isdecimal() else device)
    source = YOLO(str(weights), task='segment').model
    models = []
    for kind in ('RGB', ARM):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SEED)
            m = (FrequencySegmentationModel(source.yaml, arm='RGB', nc=21, verbose=False) if kind == 'RGB'
                 else ShapeCoeffModel(source.yaml, nc=21, verbose=False))
        m.names = dict(enumerate(names))
        m.load(source, verbose=False)
        m.args = get_cfg(overrides={'task': 'segment', 'epochs': EPOCHS, 'overlap_mask': True})
        models.append(m.to(device).eval())
    base, candidate = models
    left, right = base.state_dict(), candidate.state_dict()
    require(all(k in right and torch.equal(v, right[k]) for k,v in left.items()), 'Native initialization differs')
    added = sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in base.parameters())
    require(added == 68, 'Unexpected candidate parameter count')
    x = torch.rand((2, 3, 96, 128), generator=torch.Generator().manual_seed(19)).to(device)
    with torch.inference_mode():
        ya, ra = base(x)
        yb, rb = candidate(x)
        require(all(torch.equal(a,b) for a,b in zip(ya,yb,strict=True)), 'Zero-gain forward changed native outputs')
        for branch in ('one2many','one2one'):
            require(all(torch.equal(ra[branch][k],rb[branch][k]) for k in ('boxes','scores','mask_coefficient','proto')),
                    f'Zero-gain raw {branch} differs')
        active = deepcopy(candidate)
        for name in ('shape_one2many','shape_one2one'):
            p = getattr(active.model[-1], name)
            p.projection.bias.fill_(.3)
            p.gain.fill_(.4)
        _, edited = active(x)
        for branch in ('one2many','one2one'):
            p, q = rb[branch], edited[branch]
            require(all(torch.equal(p[k],q[k]) for k in ('boxes','mask_coefficient','proto')),
                    'Active correction changed box or mask tensors')
            other = [j for j in range(21) if j != FOCUS]
            require(torch.equal(p['scores'][:,other],q['scores'][:,other]) and
                    not torch.equal(p['scores'][:,FOCUS],q['scores'][:,FOCUS]), 'Residual is not isolated to class 7')
    with TemporaryDirectory(prefix='shape-coef-reload-') as temp:
        p = Path(temp) / 'reload.pt'
        torch.save({'model': active.cpu(), 'train_args': {'task':'segment'}}, p)
        loaded = YOLO(str(p), task='segment').model.to(device).eval()
        require(loaded.frequency_arm == ARM, 'Reload lost the shape arm')
        with torch.inference_mode():
            result = loaded(x)[1]
            require(torch.equal(result['one2one']['scores'], edited['one2one']['scores']), 'Reload changed active scores')
        loaded.fuse(verbose=False)
        require(loaded.model[-1].shape_one2many is None and loaded.model[-1].shape_one2one is not None,
                'Fusing removed the wrong shape projector')
        with torch.inference_mode():
            require(torch.isfinite(loaded(x)[0][0]).all(), 'Fused inference is not finite')

    masks = torch.zeros((2, 24, 32), dtype=torch.uint8)
    masks[:, 6:18, 8:24] = 1
    masks[1, 9:15, 17:24] = 0
    batch = {'img': x, 'batch_idx': torch.tensor([0,1]), 'cls': torch.tensor([[7.],[13.]]),
             'bboxes': torch.tensor([[.5,.5,.5,.5],[.5,.5,.5,.5]]), 'masks': masks,
             'sem_masks': masks * torch.tensor([7,13], dtype=torch.uint8)[:,None,None]}
    batch['shape_target'], batch['shape_valid'] = targets_from_masks(batch, overlap=True)
    require(batch['shape_valid'].all() and batch['shape_target'][0] > batch['shape_target'][1], 'Synthetic shape targets invalid')
    batch = {k:v.to(device) for k,v in batch.items()}
    candidate.train()
    raw = candidate(x)
    native_criterion, shape_criterion = base.init_criterion(), candidate.init_criterion()
    branches = {}
    for name in ('one2many','one2one'):
        ref, mod = getattr(native_criterion,name), getattr(shape_criterion,name)
        p = raw[name]
        assignment, losses, _ = ref.get_assigned_targets_and_loss(p,batch)
        assignment_b, corrected, _ = mod.get_assigned_targets_and_loss(p,batch)
        require(all(torch.equal(a,b) for a,b in zip(assignment,assignment_b,strict=True)), 'Assignment logic changed')
        require(torch.equal(losses[0], corrected[0]) and torch.equal(losses[2], corrected[2]) and corrected[1] > losses[1],
                'Auxiliary did not affect just the classification loss slot')
        aux = shape_regression(p['shape_readout'], assignment[0], assignment[1], batch)
        projector = getattr(candidate.model[-1], 'shape_' + name)
        gradient, mask_gradient = torch.autograd.grad(aux, (projector.projection.weight, p['mask_coefficient']),
                                                      allow_unused=True, retain_graph=True)
        require(mask_gradient is None and gradient is not None and torch.isfinite(gradient).all()
                and torch.count_nonzero(gradient) > 0, 'Shape teacher gradient leaked or did not activate')
        branches[name] = {'foreground': int(assignment[0].sum()), 'shape_loss': float(aux.detach())}
    total = shape_criterion(raw,batch)[0].sum()
    require(torch.isfinite(total), 'Full training loss is nonfinite')
    total.backward()
    require(all(p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters()), 'Full backward is nonfinite')
    return {'decision':'PASS','protocol':PROTOCOL,'device':str(device),'added_training_parameters':added,
            'added_fused_inference_parameters':34,'initial_native_outputs_exact':True,'active_changes_only_class_7':True,
            'checkpoint_reload_and_fusion_pass':True,'shape_teacher_isolated_from_mask_coefficients':True,
            'synthetic_gradients':branches,'optimizer_step_executed':False,'training_executed':False,'test_images_accessed':False}


def run_contract(arm, device):
    contract = {'protocol':PROTOCOL,'arm':arm,'seed':SEED,'epochs':EPOCHS,'imgsz':IMAGE_SIZE,'batch':BATCH,
                'pretrained_checkpoint_sha256':WEIGHTS_SHA256,'development_sha256':DEVELOPMENT_SHA,
                'runner_sha256':_sha256(Path(__file__)),
                'shape_code_sha256':_sha256(Path(__file__).parents[1] / 'auliya_shape_coeff.py'),
                'ultralytics_version':ULTRALYTICS_VERSION,'torch_version':str(torch.__version__),
                'gpu':torch.cuda.get_device_name(int(device))}
    if bundle_sha := os.environ.get('AULIYA_KAGGLE_BUNDLE_SHA256'):
        contract['kaggle_bundle_sha256'] = bundle_sha
        contract['torch_cuda_version'] = torch.version.cuda
        contract['cudnn_version'] = torch.backends.cudnn.version()
    return contract


def train(arm, development, weights, output, device):
    if not device.isdecimal() or not torch.cuda.is_available() or int(device) >= torch.cuda.device_count():
        raise RuntimeError('Training requires an available CUDA device')
    data = verify(development, weights)
    if arm == ARM:
        print(json.dumps(preflight(development, weights, device), ensure_ascii=False), flush=True)
    outputs = output.resolve() / arm
    last = outputs / 'weights' / 'last.pt'
    report = outputs / 'validation_result.json'
    marker = outputs / 'run_contract.json'
    last.parent.mkdir(parents=True, exist_ok=True)
    contract = run_contract(arm, device)
    lock = outputs / '.active_run'
    with lock.open('x', encoding='utf-8') as f:
        f.write('One active training session per arm.\n')
    try:
        if marker.exists():
            require(json.loads(marker.read_text(encoding='utf-8')) == contract, 'Existing arm contract differs')
        elif last.exists() or report.exists():
            raise RuntimeError('Existing outputs without an arm contract')
        else:
            _write_json(marker, contract)
        if report.exists():
            final = json.loads(report.read_text(encoding='utf-8'))
            require(final['run_contract'] == contract and final['checkpoint_sha256'] == _sha256(last), 'Stale validation report')
            return final
        with TemporaryDirectory(prefix=f'{arm}-fresh-') as t:
            local = Path(t) / arm
            lpt = local / 'weights' / 'last.pt'
            lpt.parent.mkdir(parents=True)
            completed = 0
            if last.exists():
                shutil.copyfile(last,lpt)
                completed,state = _load_stage(lpt,arm)
                _restore_results(local/'results.csv',state.get('train_results',{}))
            if completed < EPOCHS:
                y = YOLO(str(lpt if completed else weights),task='segment')

                def persist(trainer):
                    require(trainer.batch_size == BATCH and trainer.epochs == EPOCHS, 'Frozen training budget changed')
                    _publish_file(trainer.last,last)
                    if (local/'results.csv').is_file():
                        _publish_file(local/'results.csv',outputs/'results.csv')
                    _write_json(outputs/'progress.json',{'epoch':trainer.epoch+1,'epochs':EPOCHS,
                                                         'checkpoint_sha256':_sha256(last)})
                    print(f'{arm}: epoch {trainer.epoch+1}/{EPOCHS} saved', flush=True)

                y.add_callback('on_model_save',persist)
                y.train(trainer=ShapeCoeffTrainer if arm == ARM else _trainer_for('RGB'),
                        data=str(data),project=str(local.parent),name=arm,save_dir=str(local),
                        resume=str(lpt) if completed else False,epochs=EPOCHS,patience=0,seed=SEED,
                        imgsz=IMAGE_SIZE,batch=BATCH,workers=2,deterministic=True,device=device,plots=False,val=True)
            require(_load_stage(lpt,arm)[0] == EPOCHS, 'Fixed last checkpoint missing')
            y = YOLO(str(lpt),task='segment')
            diag = Counter()
            seen = set()
            expected = {p.resolve() for p in (data.parent/'val/images').glob('*.jpg')}

            class Capture(SegmentationValidator):
                def update_metrics(self, preds, batch):
                    for i, p in enumerate(preds):
                        truth = DetectionValidator._prepare_batch(self,i,batch)
                        path = Path(truth['im_file']).resolve()
                        if path in seen or path not in expected:
                            raise RuntimeError('Unexpected validation image')
                        seen.add(path)
                        gt, labels, conf = truth['cls'], p['cls'], p['conf']
                        ious = box_iou(truth['bboxes'],p['bboxes'])
                        j = matched_indices(ious.cpu().numpy(),conf.cpu().numpy())
                        focus = np.flatnonzero(gt.cpu().numpy() == FOCUS).tolist()
                        high = conf >= .25
                        tp = self.match_predictions(labels[high],gt,ious[:,high])[:,0]
                        diag['targets'] += len(focus)
                        diag['target_matched'] += sum(k in j for k in focus)
                        diag['target_correct'] += sum(k in j and int(labels[j[k]]) == FOCUS for k in focus)
                        diag['target_fp'] += int(((labels[high] == FOCUS) & ~tp).sum())
                        diag['all_fp'] += int((~tp).sum())
                    return super().update_metrics(preds,batch)

            v = y.val(validator=Capture,data=str(data),split='val',imgsz=IMAGE_SIZE,batch=BATCH,device=device,
                      project=str(local),conf=.001,plots=False)
            require(seen == expected and diag['targets'] == 60, 'Validation inventory or target count changed')
            box = {k.replace('mask','box'): val for k,val in _summarize_masks(v.box,v.names).items()}
            final = {'run_contract':contract,'checkpoint_sha256':_sha256(last),'box':box,
                     'mask':_summarize_masks(v.seg,v.names),'evaluation_split':'val','test_images_accessed':False}
            final['focus_at_conf025_iou050'] = dict(diag)
        _write_json(report,final)
        return final
    finally:
        lock.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--arm',choices=('RGB',ARM),default=ARM)
    p.add_argument('--device',default='cpu')
    p.add_argument('--authorize-training',action='store_true')
    a = p.parse_args()
    if a.authorize_training and a.output is None:
        p.error('--output required for training')
    result = train(a.arm,a.development,a.weights,a.output,a.device) if a.authorize_training else preflight(a.development,a.weights,a.device)
    if not a.authorize_training and a.output:
        a.output.mkdir(parents=True,exist_ok=True)
        _write_json(a.output/'static_audit.json',result)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
