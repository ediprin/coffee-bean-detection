"""Literature controls: native Copy-Paste and a restricted photometric OA-Mix port.

OA-Mix reference: Lee et al., AAAI 2024, DOI 10.1609/aaai.v38i4.28076,
https://github.com/WoojuLee24/OA-DG/tree/ba6d86e86c65fa9c315c256b79d39cd325f5083d
Not full OA-DG: no geometric local warps, no OA-Loss, no extra views.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics.data.augment import CopyPaste, Format
from coffee_detector.experiments.run_auliya_official_frequency import trainer_for as native_trainer

P=.5
ARMS=("D1","D2")
OPS=(ImageEnhance.Color,ImageEnhance.Contrast,ImageEnhance.Brightness,ImageEnhance.Sharpness)


def spectral_saliency(patch):
    """Portable spectral-residual approximation, not OpenCV contrib bit parity."""
    if min(patch.shape[:2])<4:
        return -1.
    gray=cv2.resize(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY),(64,64)).astype(np.float32)
    spec=np.fft.fft2(gray)
    logamp=np.log(np.abs(spec)+1e-8)
    residual=logamp-cv2.blur(logamp,(3,3))
    sal=np.abs(np.fft.ifft2(np.exp(residual+1j*np.angle(spec))))**2
    sal=cv2.GaussianBlur(sal,(5,5),8)
    peak=float(sal.max())
    return float((sal/peak).mean()*255) if peak>0 else 0.


def regions(h,w,scale,count,rng):
    """Nonoverlapping rectangular local regions as in OA-Mix."""
    boxes=[]
    for _ in range(50):
        if len(boxes)>=count: break
        area=rng.uniform(*scale)*h*w
        ratio=rng.uniform(1/3,3)
        bw,bh=max(1,int(np.sqrt(area/ratio))),max(1,int(np.sqrt(area*ratio)))
        if bw>w or bh>h: continue
        x,y=int(rng.randint(w-bw+1)),int(rng.randint(h-bh+1))
        b=(x,y,x+bw,y+bh)
        if any(overlap(b,a) for a in boxes): continue
        boxes.append(b)
    return boxes


def overlap(a,b):
    return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])


def soft_mask(box,h,w):
    x,y,x2,y2=box
    # Source-style low-resolution mask blur, then resize for mixing.
    hh,ww=max(1,h//4),max(1,w//4)
    mask=np.zeros((hh,ww),np.float32)
    a,b=int(x*ww/w),int(y*hh/h)
    c,d=min(ww,max(a+1,int(np.ceil(x2*ww/w)))),min(hh,max(b+1,int(np.ceil(y2*hh/h))))
    mask[b:d,a:c]=1
    mask=cv2.GaussianBlur(mask,(0,0),max((c-a)*.2,.5),sigmaY=max((d-b)*.2,.5))
    return cv2.resize(mask,(w,h))


def photo(image,rng):
    # Restricted photometric subset, bounded factor instead of source severity 10.
    op=OPS[int(rng.randint(len(OPS)))]
    return np.asarray(op(Image.fromarray(image[:,:,::-1])).enhance(float(rng.uniform(.8,1.2))))[:,:,::-1].copy()


def oa_photo(image,boxes,rng=np.random):
    """One output view, same spatial grid and annotations."""
    h,w=image.shape[:2]
    boxes=[tuple(map(int,b)) for b in boxes]
    boxes=[(max(0,x),max(0,y),min(w,x2),min(h,y2)) for x,y,x2,y2 in boxes]
    boxes=[b for b in boxes if b[2]>b[0] and b[3]>b[1]]
    local=regions(h,w,(.01,.1),int(rng.randint(1,3)),rng)
    weights=rng.dirichlet([1.,1.,1.])
    aug=np.zeros_like(image,dtype=np.float32)
    for weight in weights:
        current=image.copy()
        for _ in range(int(rng.randint(1,4))):
            out=photo(current,rng)
            for x,y,x2,y2 in local:
                out[y:y2,x:x2]=photo(current[y:y2,x:x2].copy(),rng)
            current=out
        aug+=float(weight)*current
    scores=[spectral_saliency(image[y:y2,x:x2]) for x,y,x2,y2 in boxes]
    selected=[(b,s) for b,s in zip(boxes,scores) if s<=10]
    random_boxes=regions(h,w,(.005,.1),min(max(len(selected),1),5),rng)
    for b in random_boxes:
        score=min((s for a,s in zip(boxes,scores) if overlap(a,b)),default=float("inf"))
        selected.append((b,score))
    # Convex normalized overlap blending. Source uses a different overlap reduction.
    total=np.zeros((h,w),np.float32)
    weighted=np.zeros_like(total)
    for box,score in selected:
        mask=soft_mask(box,h,w)
        total+=mask
        weighted+=mask*float(rng.uniform(0,.5 if score<=10 else 1))
    local_weight=weighted/np.maximum(total,1e-8)
    coverage=np.clip(total,0,1)
    mix=coverage*local_weight+(1-coverage)*float(rng.beta(1,1))
    return np.clip(np.rint(image*(1-mix[:,:,None])+aug*mix[:,:,None]),0,255).astype(np.uint8)


class OAPhoto:
    """Inserted before Format, after native transforms; labels untouched."""
    def __init__(self,p=P):
        self.p=p

    def __call__(self,labels):
        if np.random.random()>=self.p: return labels
        h,w=labels["img"].shape[:2]
        instances=labels["instances"]
        # Read polygon geometry without mutating the native Instances object.
        polys=np.asarray(instances.segments)
        if len(polys):
            polys=polys.copy()
            if instances.normalized: polys*=np.array([w,h])
            boxes=np.concatenate((polys.min(1),polys.max(1)),axis=1)
        else:
            boxes=np.empty((0,4))
        labels["img"]=oa_photo(labels["img"],boxes)
        return labels


class D2Trainer(native_trainer("RGB")):
    def build_dataset(self,img_path,mode="train",batch=None):
        ds=super().build_dataset(img_path,mode,batch)
        if mode=="train":
            if not isinstance(ds.transforms.transforms[-1],Format):
                raise ValueError("Expected native final Format")
            ds.transforms.insert(-1,OAPhoto())
            # Native close_mosaic rebuilds transforms and removes this addition.
        return ds


def trainer_for(arm):
    if arm=="D1": return native_trainer("RGB")
    if arm=="D2": return D2Trainer
    raise ValueError(arm)


def flatten_transforms(transform):
    yield transform
    for t in getattr(transform,"transforms",[]):
        yield from flatten_transforms(t)


def check_train_pipeline(trainer,arm):
    ts=list(flatten_transforms(trainer.train_loader.dataset.transforms))
    if arm=="D1":
        if not any(isinstance(t,CopyPaste) and t.mode=="mixup" and t.p==P for t in ts):
            raise RuntimeError("D1 cross-image CopyPaste is not active")
    else:
        if sum(isinstance(t,OAPhoto) for t in ts)!=1:
            raise RuntimeError("D2 OAPhoto missing or duplicated")
    if any(isinstance(t,OAPhoto) for t in flatten_transforms(trainer.test_loader.dataset.transforms)):
        raise RuntimeError("Unexpected validation augmentation")
    print("Verified training-only augmentation:",arm,flush=True)
