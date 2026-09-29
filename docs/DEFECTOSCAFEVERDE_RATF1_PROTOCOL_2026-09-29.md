# DefectosCafeVerde RATF1 Seed-42 Protocol

Date frozen: 2026-09-29  
Status: implementation and static verification only; training not yet executed  
Dataset: `defectoscafeverde-grouped-physical-v1`  
Test policy: locked; the development root must expose only train and validation

## Research question

Can a selective texture residual improve the remaining visually similar coffee
defect classes without replacing the strong native RGB representation or
disturbing localization?

The grouped DefectosCafeVerde baseline is already strong, while the published
error analysis concentrates residual confusion in visually similar morphology
and colour families. Earlier full-image Fourier/wavelet preprocessing and
global handcrafted-feature fusion did not improve the matched native detector.
Tulsi et al. (2026) nevertheless report that a small subset of coarse wavelet
and low-frequency oblique Gabor descriptors remains informative, while global
fusion can become redundant with learned image embeddings. RATF1 tests that
narrower hypothesis in the detector's classification branch.

## Frozen candidate

`RATF1` means Redundancy-Aware Selective Texture Fusion.

1. The detector consumes the original RGB image; no transformed image replaces
   its input.
2. A deterministic luminance-only cue contains exactly three channels:
   level-3 vertical Haar magnitude and low-frequency Gabor energy at 114 and
   172 degrees. Gabor analysis runs at half resolution with wavelength and
   kernel scaled accordingly, preserving the intended image-space scale while
   avoiding a full-resolution large-kernel cost.
3. At every detection scale, the cue is encoded and its component parallel to
   the native RGB feature is removed per spatial location.
4. The remaining orthogonal component predicts a bounded, class-specific score
   residual through a learned class gate.
5. Native box features and box heads are unchanged. The score residual is
   zero-initialized, so the complete detector must initially equal D0DIRECT.
6. No ROI crop, decoded-box dependency, router, test image, or dataset-fitted
   texture threshold is used.

This is an adaptation motivated by the cited feature findings, not a claim to
reproduce Tulsi et al.'s classification pipeline.

## Controlled comparison

- Control: the existing valid `D0DIRECT` seed-42 result from
  `defectoscafeverde-grouped-dcwcf-direct-seed42-v1`.
- Candidate: one fresh `RATF1` seed-42 run from the same official
  `yolo26n.pt` checkpoint.
- Model: YOLO26n P3, 12 classes.
- Schedule: 50 epochs, image size 640, batch 16, patience 15, optimizer auto,
  close-mosaic 10, seed 42, deterministic mode.
- Selection split: validation only.
- Primary metrics: Macro, Bottom-3, and Worst-class mAP50-95.
- Diagnostic classes fixed before training: `agrio`, `concha`, `negro`,
  `oreja`, and `partido`.

The decision code must verify equality of dataset-audit SHA, official
pretrained SHA, native config SHA, schedule, and seed between endpoints.

## Static gates

Training is forbidden unless all gates pass:

- official pretrained SHA and 12-class ontology are exact;
- native and RATF model YAML/schedule match;
- common backbone and wrapped native head tensors are exact;
- cue is deterministic, finite, nonzero, and three-channel;
- projected texture residual is numerically orthogonal to its RGB reference;
- initial raw boxes and scores are bitwise equal to D0DIRECT;
- an activated residual changes scores but preserves boxes;
- classification-residual gradients are finite and nonzero;
- no test access occurs.

## Seed-42 decision

Promote to paired confirmation if either route passes:

- overall Pareto route: Macro improves, Bottom-3 does not decrease, and Worst
  does not decrease; or
- lower-tail Pareto route: Macro decreases by at most 0.1 percentage point,
  while both Bottom-3 and Worst improve.

Otherwise stop RATF1 after seed 42. No test evaluation is authorized by this
protocol.
