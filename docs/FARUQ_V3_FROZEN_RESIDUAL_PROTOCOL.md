# Faruq-v3 Frozen-D0 Multilevel Residual Protocol

Version: `v1.0.0`

Frozen: 2026-08-02, before FRM1 initialization or training

## Evidence and question

MHF1 improved over its capacity-matched P5 control by 5.30 Macro mAP50-95
points, but remained 1.65 points below native D0. The question is therefore no
longer whether multilevel features contain signal. It is whether that signal
can correct uncertain D0 classifications without changing the already stronger
native detector.

## Model FRM1

FRM1 is one serialized YOLO26 detector, not a crop pipeline or a second
inference process:

1. initialize from the completed Faruq-v3 D0 seed-42 `best.pt`;
2. freeze the backbone, neck, native box branches, native class branches,
   buffers, and BatchNorm running statistics;
3. preserve D0 candidate generation and localization unchanged;
4. ROIAlign matched predicted boxes on P3, P4, and P5;
5. form the capacity-matched 512-dimensional multilevel descriptor already
   justified by the CM512 audit;
6. predict a leaf-class residual;
7. multiply the residual by a learned confidence gate derived only from frozen
   D0 class probabilities; and
8. add the gated residual to D0 logits before native postprocessing.

The residual classifier is initialized to exactly zero and the gate starts at
approximately 0.01. Consequently FRM1 must reproduce D0 before optimization.

## Frozen optimization

- data: leakage-safe Faruq-v3 grouped development train/validation;
- test: unavailable and locked;
- seed: 42 only;
- trainable parameters: multilevel refiner and confidence gate only;
- epochs: 10; image size: 640; batch: 16; workers: 2;
- optimizer: AdamW, learning rate 0.001, weight decay 0.0001;
- no mosaic; no detector fine-tuning;
- candidate source: frozen D0 one-to-one branch, top 500;
- target assignment: class-agnostic greedy IoU >= 0.5;
- ROI expansion: 1.0; residual inference weight: 1.0;
- loss: cross-entropy of frozen D0 logits plus gated residual;
- preservation penalty: 0.25 times squared correction on candidates D0 already
  classifies correctly;
- checkpoint is written directly to the shared Drive project and resumes from
  `last.pt`.

## Mandatory static gate

Before dataset training, verify:

- D0 native state is unchanged after injection;
- zero-initialized FRM1 output is numerically D0-identical at `rtol=0,
  atol=1e-7`, with raw native tensors bitwise equal;
- only refiner/gate parameters require gradients and enter the optimizer;
- native BatchNorm modules remain in evaluation mode when FRM1 is in training
  mode;
- residual and gate receive finite gradients;
- a nonzero residual changes class output without changing native raw boxes;
- state-dict round trip succeeds.

Static FAIL blocks training.

## Validation gate

FRM1 passes seed-42 screening only when all are true against the frozen D0
validation report:

1. Macro mAP50-95 improves by at least 0.5 point;
2. bottom-3 class mAP50-95 does not decrease;
3. worst-class mAP50-95 does not decrease by more than 1 point.

FAIL stops this final candidate without extra seeds or test. PASS authorizes a
three-seed confirmation protocol only; test remains locked.
