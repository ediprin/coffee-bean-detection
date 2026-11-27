"""Frozen P2/P3 global/regional mean/histogram factorial audit (B -> C only)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import sklearn
import torch
import ultralytics

from coffee_detector.analysis import auliya_representation_audit as a

PROTOCOL = "auliya-spatial-distribution-audit-v1"
EXPECTED_CONTENT = "12d60838ae810a2841607fd5a09afe6b77c8a48977ea2b9c6bc3a52394d55166"
LEVELS = ("P2", "P3")
CODES = [f"{level}_R{i}" for level in LEVELS for i in range(1,5)]
BASE = {
    "matched_P2_R1": [0.4087485968198877,0.23289968636107705,0.07407407407407407],
    "matched_P3_R1": [0.33731707985053017,0.1732999501616126,0.1],
    "raw_P2_R1": [0.44691387661513154,0.19545455834062914,0.07766990291262135],
    "raw_P3_R1": [0.345776377119011,0.16830236599014609,0.07228915662650602],
}
METRICS = ("macro18","hard5","worst18")
TOL = .005
SAMPLES = 64
RANK = 32


def lattice(mask, feature):
    return cv2.resize(mask.astype(np.float32),(feature.shape[2],feature.shape[1]),
                      interpolation=cv2.INTER_AREA)


def regional_masks(mask, poly):
    """Object-relative 2x2 quadrants in image coordinates; partition exactly."""
    h,w = mask.shape
    cx,cy = (poly.min(0)+poly.max(0))/2
    x = np.arange(w)[None,:]+.5 < cx
    y = np.arange(h)[:,None]+.5 < cy
    return [mask*(y & x),mask*(y & ~x),mask*(~y & x),mask*(~y & ~x)]


def sample_cells(feature, weight):
    """Deterministic systematic weighted sampling, equal sample count per instance."""
    flat = weight.ravel().astype(np.float64)
    cdf = np.cumsum(flat)
    if cdf[-1] <= 0:
        raise ValueError("Empty calibration mask")
    targets = (np.arange(SAMPLES)+.5)/SAMPLES*cdf[-1]
    indices = np.searchsorted(cdf,targets,side="right")
    return feature.reshape(feature.shape[0],-1)[:,indices].T


def calibrate(samples):
    # Shared channel boundaries across all operators, regions and validation.
    # Repeated quantiles are allowed: bins may be empty, never perturb with val.
    return np.quantile(samples,[.25,.5,.75],axis=0).T.astype(np.float32)


def weighted_hist(feature, weight, edges):
    c = feature.shape[0]
    values = feature.reshape(c,-1)
    w = weight.ravel()
    idx = w > 0
    w = w[idx]
    if not len(w):
        return np.zeros(c*4,np.float32)
    values = values[:,idx]
    h = np.empty((c,4),np.float32)
    for j in range(c):
        k = np.searchsorted(edges[j],values[j],side="right")
        h[j] = np.bincount(k,weights=w,minlength=4)/w.sum()
    return h.ravel()


def operators(feature, mask, poly, edges):
    w = lattice(mask,feature)
    region = [lattice(m,feature) for m in regional_masks(mask,poly)]
    if not np.allclose(sum(region),w,atol=2e-6,rtol=1e-5):
        raise ValueError("Regional masks do not partition global support")
    # R1 exactly reuses original implementation, including reduction order.
    r1 = a.masked_mean(feature,mask)
    r2 = weighted_hist(feature,w,edges)
    r3 = np.concatenate([(feature*m[None]).sum((1,2))/m.sum()
                         if m.sum()>0 else np.zeros(feature.shape[0],np.float32)
                         for m in region])
    r4 = np.concatenate([weighted_hist(feature,m,edges) for m in region])
    return (r1,r2,r3,r4)


class SpatialEncoder(a.FrozenFeatures):
    def maps(self, small):
        self.cache.clear()
        t = self.torch
        x = t.from_numpy(small.transpose(2,0,1).copy()).to(self.device).float()[None]/255.
        with t.inference_mode():
            self.model(x)
            p2 = self.cache["P2"][0].float().cpu().numpy()
            p3 = self.cache["pyramid"][0][0].float().cpu().numpy()
        if [p2.shape[1:],p3.shape[1:]] != [(160,160),(80,80)]:
            raise ValueError("Unexpected spatial feature shapes")
        return dict(zip(LEVELS,(p2,p3)))


def load_scene(source, member, encoder):
    rgb = a.read_image(source,member)
    rows = a.polygons(source,member,*rgb.shape[:2])
    small,scale,pad = a.letterbox(rgb)
    maps = encoder.maps(small)
    return maps, [(y,p*scale+pad) for y,p in rows]


def provenance(args):
    # This study is paired with the exact previously observed extracted input.
    if not args.bundle.is_dir():
        raise ValueError("Use the same extracted Kaggle development folder as F1/F2")
    with a.open_development(args.bundle) as z:
        dev, records = a.inventory(z)
        evidence = a.validate_extracted(z,args.metadata,dev,records)
    if evidence["development_content_sha256"] != EXPECTED_CONTENT:
        raise ValueError("Development fingerprint differs from completed feature audit")
    if a.sha(args.checkpoint) != a.RGB_SHA:
        raise ValueError("RGB checkpoint mismatch")
    return dev,records,evidence


def calibration(source, records, encoder, out):
    path = out/"histogram_edges.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            return {k:z[k] for k in LEVELS}
    samples = {k:[] for k in LEVELS}
    for i,member in enumerate(records["train"]):
        maps,rows = load_scene(source,member,encoder)
        for _,p in rows:
            mask = a.raster(p,a.SIZE,a.SIZE)
            for k,f in maps.items():
                samples[k].append(sample_cells(f,lattice(mask,f)))
        if (i+1)%50==0:
            print(f"Calibration B: {i+1}/{len(records['train'])}",flush=True)
    edges = {k:calibrate(np.concatenate(v)) for k,v in samples.items()}
    a.write_npz(path,**edges)
    a.write_json(out/"histogram_calibration.json",{
        "role":"train","group":"B","samples_per_instance":SAMPLES,
        "samples_per_level":{k:sum(len(x) for x in v) for k,v in samples.items()},
        "duplicate_adjacent_edges":{k:int((np.diff(e,axis=1)==0).sum()) for k,e in edges.items()},
        "edges_sha256":a.sha(path)})
    return edges


def evaluate(tr,va,names):
    results,preds = {},{}
    # Baseline equivalence before any other probe fitting.
    for key,expected in BASE.items():
        mode,code = key.split("_",1)
        p = a.fit_probe(tr[code],va[code],tr["y"],va["y"],RANK if mode=="matched" else None)
        preds[key] = p
        results[key] = a.metrics(va["y"],p,names)
        actual = [results[key][m] for m in METRICS]
        if max(abs(x-y) for x,y in zip(actual,expected)) > TOL:
            raise RuntimeError(f"R1 baseline equivalence failed: {key}, actual={actual}, expected={expected}")
    for mode,rank in (("matched",RANK),("raw",None)):
        for code in CODES:
            key = f"{mode}_{code}"
            if key in results:
                continue
            p = a.fit_probe(tr[code],va[code],tr["y"],va["y"],rank)
            preds[key] = p
            results[key] = a.metrics(va["y"],p,names)
    y = va["y"]
    hard = np.isin(y,[names.index(s) for s in a.HARD])
    clean = np.isin(y,a.CLEAN)
    effects,decisions = {},{}
    for level in LEVELS:
        for mode in ("matched","raw"):
            base = f"{mode}_{level}_R1"
            b = preds[base]
            rows = [results[f"{mode}_{level}_R{i}"] for i in range(1,5)]
            effects[f"{mode}_{level}"] = {}
            for m in METRICS:
                r1,r2,r3,r4 = [x[m] for x in rows]
                effects[f"{mode}_{level}"][m] = {
                    "histogram":((r2-r1)+(r4-r3))/2,
                    "regional":((r3-r1)+(r4-r2))/2,
                    "interaction":r4-r3-r2+r1}
            for i in range(2,5):
                key = f"{mode}_{level}_R{i}"
                p = preds[key]
                row = results[key]
                row["delta_vs_R1"] = {m:row[m]-rows[0][m] for m in METRICS}
                row["hard_class_delta_vs_R1"] = {s:row["f1_per_class"][s]-rows[0]["f1_per_class"][s] for s in a.HARD}
                row["paired_vs_R1"] = {
                    name:{"rescue":int((mask&(b!=y)&(p==y)).sum()),
                          "damage":int((mask&(b==y)&(p!=y)).sum()),
                          "both_wrong":int((mask&(b!=y)&(p!=y)).sum()),
                          "both_correct":int((mask&(b==y)&(p==y)).sum())}
                    for name,mask in (("clean18",clean),("hard5",hard))}
                if mode=="matched":
                    d = row["delta_vs_R1"]
                    n = sum(v>0 for v in row["hard_class_delta_vs_R1"].values())
                    decisions[key] = {"hard_classes_improved":n,
                        "passes":bool(d["macro18"]>=.02 and d["hard5"]>=.02 and
                                      d["worst18"]>=-.01 and n>=3)}
    return results,preds,effects,decisions


def run(args):
    cv2.setNumThreads(1)
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    dev,records,evidence = provenance(args)
    names = dev["names"]
    contract = {
        "protocol":PROTOCOL,"runner_sha":a.sha(__file__),"base_runner_sha":a.sha(a.__file__),
        "data_provenance":{k:v for k,v in evidence.items() if k!="files"},
        "checkpoint_sha":a.RGB_SHA,"device":args.device,
        "python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn.__version__,
        "opencv":cv2.__version__,"torch":torch.__version__,"ultralytics":ultralytics.__version__,
        "gpu":torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "rank":RANK,"bins":4,"samples_per_instance":SAMPLES,"test_images_accessed":False}
    marker = out/"contract.json"
    if marker.exists() and json.loads(marker.read_text())!=contract:
        raise ValueError("Resume contract mismatch")
    if not marker.exists() and any(out.iterdir()):
        raise ValueError("Nonempty output without run contract")
    a.write_json(marker,contract)
    a.write_json(out/"input_provenance.json",evidence)
    encoder = SpatialEncoder(args.checkpoint,args.device,names)
    shards = out/"shards"
    shards.mkdir(exist_ok=True)
    with a.open_development(args.bundle) as source:
        edges = calibration(source,records,encoder,out)
        edge_sha = a.sha(out/"histogram_edges.npz")
        cal_report = out/"histogram_calibration.json"
        if not cal_report.exists() or json.loads(cal_report.read_text())["edges_sha256"]!=edge_sha:
            raise ValueError("Calibration cache incomplete or modified; use fresh output")
        for role,images in records.items():
            for i,member in enumerate(images):
                target = shards/f"{role}_{i:04d}.npz"
                if target.exists():
                    with np.load(target,allow_pickle=False) as z:
                        if str(z["member"].item())!=member or str(z["edges_sha"].item())!=edge_sha or not set(CODES)<=set(z.files):
                            raise ValueError(f"Invalid cached shard: {target}")
                    continue
                maps,rows = load_scene(source,member,encoder)
                reps = {c:[] for c in CODES}
                for _,p in rows:
                    mask = a.raster(p,a.SIZE,a.SIZE)
                    for k,f in maps.items():
                        for j,v in enumerate(operators(f,mask,p,edges[k]),1):
                            reps[f"{k}_R{j}"].append(v)
                arrays = {k:np.stack(v).astype(np.float32) for k,v in reps.items()}
                if not all(np.isfinite(v).all() for v in arrays.values()):
                    raise ValueError("Nonfinite descriptor")
                a.write_npz(target,member=np.array(member),edges_sha=np.array(edge_sha),
                            ids=np.array([f"{member}#{j}" for j in range(len(rows))]),
                            y=np.array([y for y,_ in rows]),**arrays)
                if (i+1)%25==0 or i+1==len(images):
                    print(f"{role}: {i+1}/{len(images)}",flush=True)
    def collect(role):
        rows=[]
        for i in range(a.COUNTS[role]):
            with np.load(shards/f"{role}_{i:04d}.npz",allow_pickle=False) as z:
                rows.append({k:z[k] for k in ["ids","y",*CODES]})
        arrays={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
        if dict(Counter(map(int,arrays["y"])))!={int(k):v for k,v in dev["polygon_instances_by_class"][role].items()}:
            raise ValueError("Class count mismatch")
        if len(set(arrays["ids"]))!=len(arrays["ids"]):
            raise ValueError("Duplicate instance ids")
        a.write_npz(out/f"{role}_features.npz",**arrays)
        return arrays
    tr,va=collect("train"),collect("val")
    if set(tr["ids"])&set(va["ids"]):
        raise ValueError("Overlap in instance ids")
    results,preds,effects,decisions=evaluate(tr,va,names)
    with (out/"validation_predictions.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["instance","true_class",*preds])
        for i,id_ in enumerate(va["ids"]):
            writer.writerow([id_,int(va["y"][i]),*[int(p[i]) for p in preds.values()]])
    a.write_json(out/"report.json",{"contract":contract,"classes":names,
        "baseline_reference":BASE,"baseline_tolerance":TOL,"baseline_equivalence":"PASS",
        "results":results,"factorial_effects":effects,"screening":decisions,
        "decision":"DIAGNOSTIC_ONLY_NO_AUTO_PROMOTION"})
    for key,row in results.items():
        print(key,*(f"{m}={row[m]:.4f}" for m in METRICS),flush=True)
    print(json.dumps(decisions,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--metadata",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    run(p.parse_args())


if __name__=="__main__":
    main()
