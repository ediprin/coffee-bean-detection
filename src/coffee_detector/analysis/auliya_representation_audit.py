"""Auliya B->C frozen visual/representation diagnostic. No detector training/test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

PILOT_SHA = "b8795e189b85a13201e54a562cbe0f42fa7e0f2bfc0eb4b6d04f85d2b4cf57be"
RGB_SHA = "85b9b459fcd5079de3c27f8b897a4bd0c7160efa96bea25957c5eadd4d617ed3"
DEV_SHA = "23d653ca2e98c185a76bb7b2095d6afc6401876cbe48fd219cb42bdc7ecbd360"
COUNTS = {"train": 375, "val": 420}
HARD = ("biji_berlubang_satu", "biji_hitam_sebagian", "biji_muda",
        "kulit_tanduk_ukuran_besar", "biji_berlubang_lebih_satu")
CLEAN = list(range(18))
ALL = list(range(21))
SIZE = 640
CROP = 128
SEED = 42
V_CODES = ["V0", "V1", "V2", "V3", "V4", "V5"]
F_CODES = ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]
PROTOCOL = "auliya-discriminative-representation-audit-v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def inventory(z):
    raw = z.read("development/development_contract.json")
    if hashlib.sha256(raw).hexdigest() != DEV_SHA:
        raise ValueError("Pilot development contract hash changed")
    c = json.loads(raw)
    if c["groups"] != {"train": "B", "val": "C", "held_out": "A"} or c["images"] != COUNTS:
        raise ValueError("Not the frozen B/C development data")
    names = c["names"]
    if len(names) != 21 or len(set(names)) != 21 or not set(HARD) <= set(names):
        raise ValueError("Invalid class mapping")
    members = z.namelist()
    if len(set(members)) != len(members):
        raise ValueError("Duplicate ZIP members")
    records = {}
    for role, n in COUNTS.items():
        prefix = f"development/{role}/images/"
        images = sorted(p for p in members if p.startswith(prefix) and p.endswith(".jpg"))
        labels = {p for p in members if p.startswith(f"development/{role}/labels/") and p.endswith(".txt")}
        expected = {p.replace("/images/", "/labels/")[:-4] + ".txt" for p in images}
        if len(images) != n or labels != expected:
            raise ValueError(f"Invalid {role} image/label inventory")
        if any(Path(p).name != p[len(prefix):] for p in images):
            raise ValueError("Nested source image")
        records[role] = images
    return c, records


class DevelopmentFolder:
    """Read only B/C files from a Kaggle-extracted development directory."""

    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Expected development directory")
        self.members = {}
        contract = self.root / "development_contract.json"
        if not contract.is_file():
            raise ValueError("Missing development_contract.json")
        self._add(contract)
        if (self.root / "test").exists() or (self.root / "test.txt").exists():
            raise ValueError("Unexpected test directory/list in development")
        for role in COUNTS:
            for kind, suffix in (("images", ".jpg"), ("labels", ".txt")):
                folder = self.root / role / kind
                if not folder.is_dir():
                    raise ValueError(f"Missing development folder: {folder}")
                for p in sorted(folder.iterdir()):
                    if not p.is_file() or p.suffix != suffix:
                        raise ValueError(f"Unexpected development member: {p}")
                    self._add(p)

    def _add(self, p):
        # Reject file symlinks or nested symlink escapes.
        if p.is_symlink() or p.resolve() != p.absolute():
            raise ValueError(f"External/symlink development member: {p}")
        self.members["development/" + p.relative_to(self.root).as_posix()] = p

    def namelist(self):
        return list(self.members)

    def read(self, name):
        return self.members[name].read_bytes()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def open_development(path):
    return DevelopmentFolder(path) if path.is_dir() else ZipFile(path)


def validate_extracted(folder, metadata, contract, records):
    """Verify group membership against hash-pinned author manifest and fingerprint all B/C bytes.

    The extracted content fingerprint is NOT the original ZIP hash or proof of
    byte identity with the original archive. No group-A image/label bytes are read.
    """
    from coffee_detector.analysis.auliya_robusta_preflight import (
        METADATA_SHA256, _load_manifest, _load_folds, _read_text,
    )
    import yaml
    if metadata is None or sha(metadata) != METADATA_SHA256:
        raise ValueError("Extracted inputs require the exact official Zenodo metadata ZIP")
    with ZipFile(metadata) as z:
        manifest = _load_manifest(z)
        folds = _load_folds(z, manifest)
        names = yaml.safe_load(_read_text(z, "splits/data_v4_ujiA.yaml"))["names"]
    if names != contract["names"]:
        raise ValueError("Class mapping differs from official metadata")
    if len(folds["A"]["test"]) != 1351:
        raise ValueError("Unexpected held-out group size")
    expected = {}
    for role, group in (("train", "B"), ("val", "C")):
        expected[role] = {}
        for original, row in manifest.items():
            if row["group"] != group:
                continue
            split, _, basename = original.split("/")
            member = f"development/{role}/images/{split}__{basename}"
            expected[role][member] = row["objects"]
        if set(expected[role]) != set(records[role]):
            raise ValueError(f"Extracted {role} membership differs from the official manifest")
    # Membership is now known before opening any images.
    files = []
    hashes = {}
    for role, images in records.items():
        counts = Counter()
        for member in images:
            label = member.replace("/images/", "/labels/")[:-4] + ".txt"
            rows = polygons(folder, member, 1, 1)
            if len(rows) != expected[role][member]:
                raise ValueError(f"Instance count differs from manifest: {member}")
            counts.update(y for y, _ in rows)
            for name in (member, label):
                digest = sha(folder.members[name])
                files.append({"path":name, "sha256":digest})
                if name == member:
                    hashes.setdefault(digest, set()).add(role)
        if dict(counts) != {int(k):v for k,v in contract["polygon_instances_by_class"][role].items()}:
            raise ValueError(f"Extracted class counts differ: {role}")
    cross_split = sum(len(v) > 1 for v in hashes.values())
    if cross_split:
        raise ValueError(f"Exact duplicate image bytes across B/C: {cross_split}")
    files.sort(key=lambda x:x["path"])
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",",":")).encode()).hexdigest()
    return {
        "input_mode":"EXTRACTED_MANIFEST_VERIFIED",
        "original_pilot_zip_sha256_verified":False,
        "expected_original_pilot_zip_sha256":PILOT_SHA,
        "development_contract_sha256":DEV_SHA,
        "metadata_sha256":METADATA_SHA256,
        "development_content_sha256":fingerprint,
        "cross_split_exact_duplicate_hashes":cross_split,
        "files":files,
    }


def read_image(z, member):
    raw = z.read(member)
    with Image.open(io.BytesIO(raw)) as im:
        if im.format != "JPEG" or sorted(im.size) != [3000, 4000]:
            raise ValueError(f"Unexpected original geometry: {member}")
        if im.getexif().get(274, 1) != 1:
            raise ValueError(f"EXIF orientation requires manual polygon review: {member}")
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or sorted(bgr.shape[:2]) != [3000, 4000]:
        raise ValueError(f"Decode failed: {member}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def polygons(z, member, h, w):
    path = member.replace("/images/", "/labels/")[:-4] + ".txt"
    rows = []
    for line in z.read(path).decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        f = line.split()
        y = int(f[0])
        p = np.array(f[1:], dtype=np.float32).reshape(-1, 2)
        if y not in ALL or len(p) < 3 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError(f"Invalid polygon: {path}")
        p *= [w, h]
        rows.append((y, p))
    if not rows:
        raise ValueError(f"No polygons: {path}")
    return rows


def letterbox(rgb):
    h, w = rgb.shape[:2]
    scale = min(SIZE/h, SIZE/w)
    nw, nh = round(w*scale), round(h*scale)
    left, top = round((SIZE-nw)/2-.1), round((SIZE-nh)/2-.1)
    small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = cv2.copyMakeBorder(small, top, SIZE-nh-top, left, SIZE-nw-left,
                             cv2.BORDER_CONSTANT, value=(114,114,114))
    return out, np.array([nw/w, nh/h], np.float32), np.array([left,top], np.float32)


def bounds(poly, h, w):
    x0, y0 = np.floor(poly.min(0)).astype(int)
    x1, y1 = np.ceil(poly.max(0)).astype(int) + 1
    return max(0,x0), max(0,y0), min(w,x1), min(h,y1)


def raster(poly, h, w):
    mask = np.zeros((h,w), np.uint8)
    cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 1)
    return mask


def masked_crop(rgb, poly):
    h,w = rgb.shape[:2]
    x0,y0,x1,y1 = bounds(poly,h,w)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Degenerate crop")
    m = raster(poly-[x0,y0], y1-y0, x1-x0).astype(np.float32)
    # Resize premultiplied pixels and mask separately; no background colour in crop.
    a = rgb[y0:y1,x0:x1].astype(np.float32)/255.
    weight = cv2.resize(m, (CROP,CROP), interpolation=cv2.INTER_AREA)
    pixels = cv2.resize(a*m[...,None], (CROP,CROP), interpolation=cv2.INTER_AREA)
    pixels /= np.maximum(weight[...,None], 1e-6)
    if weight.sum() <= 0:
        raise ValueError("Empty crop mask")
    return np.clip(pixels,0,1), weight


def hist(x, m, lo, hi, bins=16):
    counts = np.histogram(x, bins=bins, range=(lo,hi), weights=m)[0]
    return counts / max(float(counts.sum()), 1e-8)


def visual_descriptors(rgb, mask):
    # All colour and texture operators act on an identical 128x128 normalized crop.
    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)
    colour = np.concatenate([hist(lab[:,:,i], mask, a,b)
                             for i,(a,b) in enumerate(((0,100.001),(-128,128),(-128,128)))])
    gray = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)
    inner = cv2.erode((mask > .99).astype(np.uint8), np.ones((13,13),np.uint8)).astype(bool)
    if not inner.any():
        # This is an explicit missing-support flag, not an unmasked texture estimate.
        texture = np.zeros(32 + 16 + 4 + 1, np.float32)
    else:
        # Rotation-sensitive Gabor energy on four orientations/two wavelengths.
        g = []
        for wave in (4.,8.):
            for theta in (0., np.pi/4, np.pi/2, 3*np.pi/4):
                for phase in (0., np.pi/2):
                    kernel = cv2.getGaborKernel((13,13), 3., theta, wave, .5, phase, ktype=cv2.CV_32F)
                    r = np.abs(cv2.filter2D(gray,cv2.CV_32F,kernel))[inner]
                    g.extend([r.mean(),r.std()])
        # Uniformly quantized 8-neighbour LBP (16 bins), with interior only.
        lbp = np.zeros_like(gray, dtype=np.uint8)
        for i,(dy,dx) in enumerate(((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))):
            lbp |= ((np.roll(gray,(dy,dx),(0,1)) >= gray).astype(np.uint8) << i)
        lb = hist(lbp[inner], np.ones(int(inner.sum())), 0,256,16)
        haar = []
        for axis in (0,1):
            detail = np.abs(gray-np.roll(gray,1,axis))[inner]
            haar.extend([detail.mean(),detail.std()])
        texture = np.r_[g,lb,haar,1.]
    # Fixed coarse RGB spatial descriptor, masked means per 8x8 grid.
    spatial = cv2.resize(rgb*mask[...,None],(8,8),interpolation=cv2.INTER_AREA)
    denom = cv2.resize(mask,(8,8),interpolation=cv2.INTER_AREA)
    spatial /= np.maximum(denom[...,None],1e-6)
    return colour, texture, spatial.ravel()


def shape_descriptor(poly,h,w):
    cnt = poly.astype(np.float32).reshape(-1,1,2)
    area = abs(cv2.contourArea(cnt))
    peri = cv2.arcLength(cnt, True)
    ww,hh = np.ptp(poly,axis=0)
    hull = abs(cv2.contourArea(cv2.convexHull(cnt)))
    (_, _),(rw,rh),_ = cv2.minAreaRect(cnt)
    eps = 1e-8
    # Includes image-relative scale; not a measurement in millimetres.
    hu = cv2.HuMoments(cv2.moments(cnt)).ravel()
    hu = np.sign(hu)*np.log1p(np.abs(hu)*1e6)
    return np.r_[area/(h*w),ww/w,hh/h, min(rw,rh)/(max(rw,rh)+eps),
                 4*np.pi*area/(peri**2+eps), area/(hull+eps),
                 area/(ww*hh+eps), hu].astype(np.float32)


def visual(rgb, small, p, pp):
    col,tex,sp = visual_descriptors(*masked_crop(rgb,p))
    c1,t1,s1 = visual_descriptors(*masked_crop(small,pp))
    geom = shape_descriptor(p,*rgb.shape[:2])
    return dict(zip(V_CODES, (np.r_[col,tex,sp],np.r_[c1,t1,s1],col,tex,geom,np.r_[col,tex,geom])))


def box_iou(a,b):
    lo = np.maximum(a[:,None,:2], b[None,:,:2])
    hi = np.minimum(a[:,None,2:], b[None,:,2:])
    inter = np.maximum(hi-lo,0).prod(-1)
    aa = np.maximum(a[:,2:]-a[:,:2],0).prod(-1)
    bb = np.maximum(b[:,2:]-b[:,:2],0).prod(-1)
    return inter / np.maximum(aa[:,None]+bb[None,:]-inter,1e-8)


def match_raw(gt, boxes):
    """Class-agnostic greedy one-to-one by IoU, all anchors, no score filtering."""
    ious = box_iou(gt,boxes)
    indices = np.full(len(gt),-1,dtype=int)
    used = set()
    for flat in np.argsort(-ious.ravel(),kind="stable"):
        g,p = divmod(int(flat),len(boxes))
        if indices[g] < 0 and p not in used:
            indices[g] = p
            used.add(p)
            if (indices >= 0).all():
                break
    if (indices < 0).any():
        raise ValueError("Not enough raw anchors")
    return indices, ious[np.arange(len(gt)),indices]


def masked_mean(feature, mask):
    # Area-resample mask to feature lattice; receptive-field mixing is not removed.
    m = cv2.resize(mask.astype(np.float32),(feature.shape[2],feature.shape[1]),interpolation=cv2.INTER_AREA)
    if m.sum() <= 0:
        raise ValueError("Empty feature mask")
    return (feature*m[None]).sum((1,2))/m.sum()


class FrozenFeatures:
    def __init__(self, checkpoint, device, names):
        import torch
        import ultralytics
        from ultralytics import YOLO
        if ultralytics.__version__ != "8.4.96":
            raise ValueError("Require Ultralytics 8.4.96")
        if sha(checkpoint) != RGB_SHA:
            raise ValueError("Checkpoint not the verified RGB B/C last.pt")
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("GPU not available")
        # Import custom module before loading the trusted, hash-pinned checkpoint.
        import coffee_detector.auliya_frequency_model
        wrapper = YOLO(str(checkpoint),task="segment")
        ck = wrapper.ckpt
        m = wrapper.model
        if getattr(m,"frequency_arm",None) != "RGB" or ck["epoch"] != 49:
            raise ValueError("Wrong model arm or endpoint")
        args = ck["train_args"]
        if any(args.get(k) != v for k,v in {"epochs":50,"batch":8,"imgsz":640,"seed":42}.items()):
            raise ValueError("Checkpoint training contract mismatch")
        if [m.names[i] for i in ALL] != names:
            raise ValueError("Model class order differs")
        self.model = m.to(self.device).float().eval().requires_grad_(False)
        self.head = m.model[-1]
        if list(self.head.f) != [16,19,22] or [int(x) for x in self.head.stride] != [8,16,32]:
            raise ValueError("Unexpected YOLO26n-Seg feature layout")
        self.cache = {}
        m.model[2].register_forward_hook(lambda mod, inp, out: self.cache.update(P2=out.detach()))
        self.head.register_forward_pre_hook(lambda mod, inp: self.cache.update(pyramid=tuple(x.detach() for x in inp[0])))
        for i,b in enumerate(self.head.one2one_cv3):
            b[-1].register_forward_pre_hook(lambda mod, inp, i=i: self.cache.update({f"C{i}":inp[0].detach()}))

    def __call__(self, small, polys):
        t = self.torch
        self.cache.clear()
        x = t.from_numpy(small.transpose(2,0,1).copy()).to(self.device).float()[None]/255.
        with t.inference_mode():
            result = self.model(x)
            raw = result[1]["one2one"]
            boxes = self.head._get_decode_boxes(raw)[0].T.float().cpu().numpy()
            logits = raw["scores"][0].T.float().cpu().numpy()
            p2 = self.cache["P2"][0].float().cpu().numpy()
            levels = [v[0].float().cpu().numpy() for v in self.cache["pyramid"]]
            cls = [self.cache[f"C{i}"][0].float().cpu().numpy() for i in range(3)]
        if [p2.shape[1:]]+[v.shape[1:] for v in levels] != [(160,160),(80,80),(40,40),(20,20)]:
            raise ValueError("Unexpected feature resolutions")
        # Native F0 uses a class-agnostic matched anchor's pre-logit descriptor.
        native = np.concatenate([f.reshape(f.shape[0],-1).T for f in cls],axis=0)
        if len(native) != len(boxes) or len(logits) != len(boxes):
            raise ValueError("Anchor order/shape mismatch")
        gt = np.array([np.r_[p.min(0),p.max(0)] for p in polys])
        anchors, overlaps = match_raw(gt,boxes)
        result = []
        for p,a,iou in zip(polys,anchors,overlaps,strict=True):
            mask = raster(p,SIZE,SIZE)
            f = [masked_mean(x,mask) for x in [p2,*levels]]
            result.append((dict(zip(F_CODES,[native[a],*f,np.concatenate(f[1:]),np.concatenate(f)])),
                           float(iou),logits[a],int(a)))
        return result


def metrics(y,p,names):
    f = f1_score(y,p,labels=ALL,average=None,zero_division=0)
    support = np.bincount(y,minlength=21)
    hard = [names.index(s) for s in HARD]
    return {"macro18":float(f[CLEAN].mean()), "macro21":float(f.mean()),
            "hard5":float(f[hard].mean()),"worst18":float(f[CLEAN].min()),
            "f1_per_class":dict(zip(names,map(float,f))),"support":dict(zip(names,map(int,support))),
            "confusion_true_rows":confusion_matrix(y,p,labels=ALL).tolist()}


def fit_probe(train,val,yt,yv,rank):
    # No validation-fitted preprocessing or hyperparameters.
    scale = StandardScaler().fit(train)
    a,b = scale.transform(train),scale.transform(val)
    if rank is not None:
        if min(a.shape)<rank:
            raise ValueError("Fixed PCA rank unavailable")
        pca = PCA(n_components=rank,svd_solver="full").fit(a)
        a,b = pca.transform(a),pca.transform(b)
    clf = LogisticRegression(C=1.,class_weight="balanced",max_iter=2000,random_state=SEED)
    clf.fit(a,yt)
    if (clf.n_iter_ >= clf.max_iter).any():
        raise RuntimeError("Probe did not converge")
    return clf.predict(b)


def write_npz(path, **arrays):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


def run(args):
    cv2.setNumThreads(1)
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    extracted = args.bundle.is_dir()
    if not extracted and sha(args.bundle) != PILOT_SHA:
        raise ValueError("Expected exact original B/C pilot Colab bundle")
    with open_development(args.bundle) as source:
        dev, records = inventory(source)
        evidence = (validate_extracted(source, getattr(args,"metadata",None), dev, records)
                    if extracted else {"input_mode":"ORIGINAL_ZIP_VERIFIED",
                                       "original_pilot_zip_sha256_verified":True,
                                       "expected_original_pilot_zip_sha256":PILOT_SHA})
    print("Input provenance:", json.dumps({k:v for k,v in evidence.items() if k!="files"}), flush=True)
    codes = V_CODES if args.stage=="visual" else F_CODES
    contract = {"protocol":PROTOCOL,"stage":args.stage,"bundle_sha":None if extracted else PILOT_SHA,
                "data_provenance":{k:v for k,v in evidence.items() if k!="files"},
                "checkpoint_sha":RGB_SHA if args.stage=="features" else None,
                "runner_sha":sha(__file__),"python":platform.python_version(),
                "numpy":np.__version__,"sklearn":sklearn.__version__,"opencv":cv2.__version__,
                "device":args.device if args.stage=="features" else "cpu",
                "test_images_accessed":False}
    encoder = None
    if args.stage=="features":
        import torch, ultralytics
        contract.update(torch=torch.__version__,ultralytics=ultralytics.__version__,
                        gpu=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None)
        if args.checkpoint is None or sha(args.checkpoint)!=RGB_SHA:
            raise ValueError("Verified RGB checkpoint is mandatory")
    marker = out/"contract.json"
    if marker.exists() and json.loads(marker.read_text())!=contract:
        raise ValueError("Existing output has a different contract")
    if not marker.exists() and any(out.iterdir()):
        raise ValueError("Nonempty output without run contract")
    write_json(marker,contract)
    write_json(out/"input_provenance.json",evidence)
    shard_dir = out/"shards"
    shard_dir.mkdir(exist_ok=True)
    with open_development(args.bundle) as z:
        dev, records = inventory(z)
        names = dev["names"]
        if args.stage=="features":
            encoder = FrozenFeatures(args.checkpoint,args.device,names)
        for role, images in records.items():
            for k, member in enumerate(images):
                target = shard_dir/f"{role}_{k:04d}.npz"
                if target.exists():
                    with np.load(target,allow_pickle=False) as a:
                        if str(a["member"].item())!=member or not set(codes)<=set(a.files):
                            raise ValueError(f"Invalid cached shard {target}")
                    continue
                rgb = read_image(z,member)
                rows = polygons(z,member,*rgb.shape[:2])
                small,scale,pad = letterbox(rgb)
                pp = [p*scale+pad for _,p in rows]
                meta = {"member":np.array(member),"y":np.array([y for y,_ in rows]),
                        "ids":np.array([f"{member}#{i}" for i in range(len(rows))])}
                if encoder:
                    items=encoder(small,pp)
                    reps=[r[0] for r in items]
                    meta.update(iou=np.array([r[1] for r in items]),logits=np.stack([r[2] for r in items]),
                                anchor=np.array([r[3] for r in items]))
                else:
                    reps=[visual(rgb,small,p,q) for (_,p),q in zip(rows,pp,strict=True)]
                arrays={c:np.stack([r[c] for r in reps]).astype(np.float32) for c in codes}
                if not all(np.isfinite(v).all() for v in arrays.values()):
                    raise ValueError("Non-finite descriptor")
                write_npz(target,**meta,**arrays)
                if (k+1)%25==0 or k+1==len(images):
                    print(f"{args.stage}/{role}: {k+1}/{len(images)}",flush=True)
    def collect(role):
        shards = [shard_dir/f"{role}_{k:04d}.npz" for k in range(COUNTS[role])]
        rows = []
        for p in shards:
            with np.load(p,allow_pickle=False) as a:
                rows.append({k:a[k] for k in a.files if k!="member"})
        arrays = {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
        actual = Counter(map(int,arrays["y"]))
        expect = {int(k):v for k,v in dev["polygon_instances_by_class"][role].items()}
        if dict(actual)!=expect or len(set(arrays["ids"]))!=len(arrays["ids"]):
            raise ValueError(f"Instance contract differs: {role}")
        write_npz(out/f"{role}_features.npz",**arrays)
        return arrays
    tr,va=collect("train"),collect("val")
    if set(tr["ids"]) & set(va["ids"]):
        raise ValueError("Train/validation overlap")
    results={}
    predictions={}
    # Fixed matched ranks: visual geometry has 14 dimensions; deep rank 32.
    rank=8 if args.stage=="visual" else 32
    for mode,r in (("matched",rank),("raw",None)):
        for c in codes:
            key=f"{mode}_{c}"
            pred=fit_probe(tr[c],va[c],tr["y"],va["y"],r)
            predictions[key]=pred
            results[key]=metrics(va["y"],pred,names)
    y=va["y"]
    ref="matched_"+codes[0]
    for k,p in predictions.items():
        b=predictions[ref]
        results[k]["paired_vs_"+ref]={"rescue":int(((b!=y)&(p==y)).sum()),
                                      "damage":int(((b==y)&(p!=y)).sum()),
                                      "persistent_wrong":int(((b!=y)&(p!=y)).sum())}
    if encoder:
        native=va["logits"].argmax(1)
        predictions["native_raw_argmax"]=native
        matched=va["iou"]>=.5
        results["raw_anchor_coverage"]={"matched_at_0.5":int(matched.sum()),"total":len(matched)}
        results["native_raw_argmax"]=metrics(y,native,names)
        # All-instance primary probe and localized-only sensitivity; report support.
        results["localized_subset"]={k:metrics(y[matched],p[matched],names) for k,p in predictions.items()}
    with (out/"validation_predictions.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["instance","true_class",*predictions]+(["raw_iou","raw_anchor"] if encoder else []))
        for i,id_ in enumerate(va["ids"]):
            writer.writerow([id_,int(y[i]),*[int(p[i]) for p in predictions.values()]]+
                            ([float(va["iou"][i]),int(va["anchor"][i])] if encoder else []))
    write_json(out/"report.json",{"contract":contract,"classes":names,"primary":"matched_macro18",
                                   "results":results,"decision":"DIAGNOSTIC_ONLY_NO_AUTO_PROMOTION"})
    print(json.dumps({k:{m:v[m] for m in ("macro18","hard5","worst18")} for k,v in results.items()
                      if "macro18" in v},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True,help="Exact pilot ZIP or extracted development directory")
    p.add_argument("--metadata",type=Path,help="Official metadata ZIP required for extracted input")
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--stage",choices=("visual","features"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    run(p.parse_args())


if __name__=="__main__":
    main()
