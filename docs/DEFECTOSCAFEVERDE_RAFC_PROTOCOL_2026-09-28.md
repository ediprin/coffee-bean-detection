# DefectosCafeVerde RAFC1 Seed-42 Screening Protocol

Frozen: **2026-09-28**

## Research question

Can a training-only, raw-anchored angular Fourier consistency objective improve
fine-grained defect detection without changing the native YOLO26n inference
path or directly regularizing localization?

This is a new question after two relevant negative results on the same grouped
development dataset. `DCWCF1` slightly underperformed its matched native arm,
and the learned LIF-RPF gates remained near zero; neither result supports
another permanent frequency feature injection at inference.

## Arms and evidence boundary

| Code | Training | Inference |
|---|---|---|
| `D0DIRECT` | ordinary native detection | native RGB YOLO26n |
| `RAFC1` | native raw detection plus classification consistency against a Fourier view | exactly the native RGB YOLO26n graph |

The existing `D0DIRECT` seed-42 result from protocol
`defectoscafeverde-grouped-cwcf-direct-seed42-v1` is reused only after its
dataset-audit SHA, official-checkpoint SHA, initialization, schedule, seed, and
test lock match. `RAFC1` starts fresh from the same official `yolo26n.pt`; it
does not warm-start from AF2, CWCF, LIF-RPF, or any coffee checkpoint.

## RAFC1 mechanism

For each train batch, the raw image remains the primary view. A second view is
formed by replacing a weak fraction of low-frequency amplitude with amplitude
from a cyclic batch donor. The soft mask covers one orientation sector modulo
180 degrees, is centrosymmetric, and excludes the exact DC coefficient. Source
phase is retained. There is no crop, affine transform, or box modification.

The objective is

```text
L = L_detection(raw) + 0.05 * L_KL_class(Fourier || stopgrad(raw))
```

Class logits are averaged per matched `(image, GT index)` using native YOLO
assignments. The Fourier branch contributes no native box, DFL, or detection
loss. During validation and inference the second view and consistency loss do
not exist.

The mechanism is motivated by the amplitude/style and phase/structure split in
[FACT (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.html)
and frequency-domain generalization for detectors in
[Wang et al. (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Generalized_UAV_Object_Detection_via_Frequency_Domain_Disentanglement_CVPR_2023_paper.html).
These papers motivate the frozen test; they do not establish improvement on
DefectosCafeVerde.

## Dataset, schedule, and locks

- Dataset: `defectoscafeverde-grouped-physical-v1`.
- Development root: 2,827 train and 808 validation images, 12 classes.
- Physical-bean groups do not cross splits.
- The 403-image test split must be absent from the runtime root.
- Seed 42, 50 epochs, image size 640, batch 16, patience 15, optimizer `auto`.
- Model selection and the decision use validation only.
- No hyperparameter search is authorized by this protocol.

## Static gates

Before training, the audit must establish:

- exact official pretrained SHA and identical target initialization;
- identical model YAML and training schedule to `D0DIRECT`;
- identical state keys, state tensors, parameter count, and raw inference
  output before training;
- an active, finite, deterministic-under-fixed-RNG Fourier transform;
- finite nonzero classification-consistency gradients;
- zero trainable or inference parameters added;
- no ROI, decoded-box dependency, or test access.

## Seed-42 decision

This is a one-arm validation screen, not a superiority claim. Promotion is
allowed by either route:

1. Pareto route: Macro improves, Bottom-3 does not fall, and Worst falls by no
   more than 0.5 point.
2. Tail route: Macro falls by no more than 0.2 point while Bottom-3 and Worst
   each improve by at least 0.5 point.

If neither route passes, stop RAFC1 without additional seeds or test access.
If it passes, first run a matched non-Fourier paired-view control before any
multi-seed claim, so a gain is not automatically attributed to Fourier.
