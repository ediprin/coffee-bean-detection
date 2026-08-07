# Faruq-v3 SAFPN/SAFM Classification-Alignment Search Protocol

Date frozen: 2026-08-07
Stage: broad candidate search
Candidate: `SAF1`
Branch: `agent/safpn-classification-alignment`

## Research question

Does learned adjacent-scale spatial alignment recover fine-grained class signal
that is present across P3/P4/P5 but is not fully exploited by previous naive or
field-level fusion heads?

This question is motivated by the repository's own diagnostics: capacity-matched
P3+P4+P5 descriptors substantially outperformed P5-only descriptors, while prior
integrated multilevel heads did not reliably beat native YOLO26. The experiment
therefore tests spatial alignment as a distinct mechanism rather than merely
repeating multilevel concatenation.

## Paper-derived operator

The implementation is based on the Spatial-Aware Alignment Fusion Module (SAFM)
in Li, Chen, and Li, *IEEE TGRS*, 2025, equations (3)-(8):

1. concatenate the upsampled deep feature and the adjacent shallow feature;
2. predict two 2-D offset maps with separate 1x1 convolutions;
3. bilinearly align both feature maps using the learned offsets;
4. obtain one spatial weight map from channel-average and channel-max pooled
   concatenated features;
5. fuse the weighted aligned features while retaining the original unwarped
   features as priors.

The paper explicitly retains the original features because the offset maps have
no direct supervision and may be uncertain. The repository implementation keeps
that safeguard.

## YOLO26 adaptation boundary

This is **not claimed to be a literal reproduction of the complete ORCNN
SAFPN**. YOLO26 P3/P4/P5 have unequal channels and already emerge from a neck.
For a clean causal screen:

- the deep feature is projected by 1x1 convolution to the adjacent shallow
  channel count before SAFM;
- alignment is performed top-down P5->P4, then aligned P4->P3;
- native YOLO26 box/regression heads consume the untouched original features;
- aligned P3/P4 representations feed zero-initialized residual class
  corrections only;
- P5 classification remains native;
- no ROIAlign, decoded-box crop, candidate top-k, or test-set access is used.

Zero-initialized final class corrections guarantee that a freshly injected SAF1
model produces the same native D0 predictions before learning.

## Fixed training setup

- data: leakage-safe Faruq-v3 grouped development train/val only;
- test split: unavailable and must remain absent;
- seed: 42 for broad search;
- initialization: frozen D0 seed-42 checkpoint;
- epochs: 50;
- imgsz: 640;
- batch: 16;
- patience: 15;
- optimizer: Ultralytics `auto`;
- close mosaic: 10;
- evaluation: validation only;
- comparators: existing D0FT seed42 and ACMC1 seed42 results.

## Frozen broad-search retention gate

This stage is deliberately a discovery screen, not confirmation. A candidate is
`RETAIN` only when it has at least one nontrivial signal against D0FT:

- Macro mAP50-95 gain >= +0.20 percentage points, **or**
- Bottom-3 mAP50-95 gain >= +0.50 points, **or**
- Worst-class mAP50-95 gain >= +0.50 points,

and all safeguards hold:

- Macro drop no worse than -1.00 point;
- Bottom-3 drop no worse than -2.00 points;
- Worst-class drop no worse than -2.00 points.

`RETAIN` only places SAF1 in the candidate pool. It does not authorize a final
test evaluation and does not imply superiority to ACMC1.

## Required pre-training contract

`tests/test_safpn_alignment.py` must pass and demonstrate:

- SAFM tensor/offset contracts;
- zero-offset initialization;
- fresh SAF1 reproduces native D0 predictions before learning;
- active SAF1 changes class scores but preserves box tensors;
- gradients can reach the learned offsets after the zero-initialized residual
  classifier becomes active;
- fused inference preserves the YOLO26 one-to-one output contract;
- no ROIAlign, top-k candidate selection, or decoded-box dependency is present
  in the alignment head.

## Decision boundary

- `RETAIN`: keep SAF1 for later confirmation/composition search.
- `REJECT`: archive SAF1 and continue broad search with APCL, DC2-style raw-crop
  re-encoding, DRNet-style refinement, DSRDet-style regularization, and other
  mechanistically distinct candidates.
- Never open the locked test split during broad search.
