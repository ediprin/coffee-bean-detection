# Faruq-v3 FMH1 Focal-Modulation Protocol

Status: frozen before training. Test remains locked.

## Research question

Can local-to-global focal modulation improve the fine-grained classification
path of YOLO26n on the 21-class Faruq-v3 SNI task while leaving localization
native and outperforming the retained STB1 classification head?

This is not a claim that focal modulation is new. The operator follows the
official Microsoft FocalNet implementation from Yang et al., NeurIPS 2022.
Muchtar et al. (IEEE Access, 2025) already evaluated a FocalNet classifier on a
four-class coffee dataset. The experiment here tests a narrower transfer gap:
classification-only focal modulation inside a one-stage 21-class SNI detector.

## Frozen model

- `D0`: native YOLO26n source checkpoint.
- `STB1`: strongest canonical breadth candidate (88.67 Macro, 83.64 Bottom-3,
  80.81 Worst on validation, seed 42).
- `FCT0`: STB optimization-matched continuation control (89.40, 84.83, 84.15).
- `FMH1`: two official-style FocalNet blocks on each P3/P4/P5 classification
  input. Nested depth-wise kernels are 3 and 5, followed by gated global
  context. The native box branches receive the original features unchanged.

The two-block depth is matched to STB1's unshifted-plus-shifted two-block
depth. A scalar gate initialized to zero makes FMH1 bitwise identical to D0 at
initialization; it is a safe initialization device, not attributed to FocalNet.

## Dataset and execution

- Faruq-v3 grouped development dataset only.
- Seed 42, 50 epochs, image size 640, batch 16, same native YOLO schedule as
  STB1.
- Validation only. The test archive must not be extracted.
- Output and resumable checkpoints live under the single shared Drive project.

## Static gate

Training is forbidden unless the audit proves:

1. strict native D0 head transfer;
2. zero-gate boxes, scores, and final output are bitwise identical to D0;
3. active focal modulation changes class scores but not raw boxes;
4. gradients reach focal-modulation parameters;
5. exactly P3/P4/P5 are modulated and no ROIAlign, candidate top-k, or box
   decoding occurs before classification;
6. nested kernels are the paper defaults 3 and 5.

## Seed-42 decision gate

`STB1 vs FMH1` passes only if FMH1 gains at least 0.5 Macro point, does not
lower Bottom-3, and loses no more than 1 Worst-class point.

The stricter `FCT0 vs FMH1` control passes only if FMH1 is not lower in Macro
or Bottom-3 and loses no more than 1 Worst-class point. FMH1 is retained only
if both comparisons pass. Failure stops FMH1 without extra seeds or test
access. Passing authorizes a separate capacity/optimization control, not an
immediate test evaluation.

## Primary sources

- Yang et al., *Focal Modulation Networks*, NeurIPS 2022:
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/1b08f585b0171b74d1401a5195e986f1-Abstract-Conference.html
- Official implementation: https://github.com/microsoft/FocalNet
- Muchtar et al., *Edge AI-Based Detection for Defective Coffee Beans Using
  Deep Learning*, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3561189.
