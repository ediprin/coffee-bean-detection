# Coffee Standard J25 — YOLOv8n + CWCF1 Portability Screen

Status: **frozen before training**  
Date: 2026-09-20

## Research question

Does the already-frozen CWCF1 mechanism improve the strong matched YOLOv8n
baseline when all data, augmentation, training-budget, and seed conditions are
held fixed?

This is **not** a new CWCF variant. The chromatic-wavelet cue, attribute
factorization, auxiliary gain, and classification-only conditioning are kept
identical to the original `CWCF1`.

## Motivation

The matched seed-42 controls established that:

- `V8N_MATCHED` is substantially stronger than the current YOLO26n controls;
- therefore the next CWCF question is portability to a stronger native detector,
  not additional tuning of the YOLO26n-specific result;
- YOLOv8n is selected first because it keeps the nano/lightweight scale while
  providing the strongest lower-tail result among the matched YOLOv8 seed-42
  screens.

The scientific comparison is:

```
V8N_MATCHED  vs  V8N_CWCF1
```

not YOLOv8n versus YOLO26n.

## Frozen CWCF mechanism

The following values must match `CWCF1.yaml` exactly:

- cue channels: 4;
- attribute gain: 0.15;
- wavelet levels: 2;
- cue clip: 4.0;
- no conditional-confusion loss;
- no repeat sampler;
- no explicit-composition gate.

The cue remains:

1. luminance-reduced blue chroma;
2. luminance-reduced red chroma;
3. level-1 Haar detail energy;
4. level-2 Haar detail energy.

The native RGB detector continues to receive RGB input. The cue modifies only
classification features through zero-initialized affine residual adapters.
Box-regression features remain native.

## YOLOv8n architecture pin

The candidate uses
`configs/coffee_fg/models/yolov8n-p3.yaml`, pinned from the Ultralytics
8.4.96 YOLOv8 P3/8-P5/32 architecture with scale fixed to `n`.

Before training, the preflight must prove that this pinned YAML reconstructs the
same native YOLOv8n detector state as the architecture stored in
`yolov8n.pt` when both are initialized with the frozen seed and loaded from
the same pretrained source.

## Matched baseline contract

`V8N_CWCF1` must match `V8N_MATCHED` on:

- dataset: `coffee-standard-j25-train-siblings-v2`;
- source-level split and exact development-contract hashes;
- official/pretrained `yolov8n.pt` bytes used by the baseline run;
- seed 42;
- 50 epochs;
- image size 640;
- batch size 16;
- patience 15;
- optimizer `auto`;
- semantic-safe augmentation dictionary;
- no repeat sampler;
- locked test remains unextracted.

## Frozen preflight gates

Training is authorized only if all gates pass:

1. the stored `V8N_MATCHED` result has the expected protocol and seed;
2. pretrained checkpoint SHA256 exactly matches the one recorded by
   `V8N_MATCHED`;
3. candidate training schedule equals `V8N_MATCHED`;
4. CWCF hyperparameters equal the original `CWCF1`;
5. pinned YOLOv8n YAML and checkpoint-native YAML produce identical native
   state tensors and bitwise-identical raw boxes/scores;
6. before activating CWCF, candidate and native YOLOv8n raw boxes and scores are
   bitwise identical;
7. after activating a probe cue residual, class scores change while boxes remain
   bitwise identical;
8. the fixed J25 black/broken attribute conjunction is preserved;
9. auxiliary gradients are finite and nonzero;
10. locked test is not accessed.

## Seed-42 reporting

Report:

- Macro mAP50–95;
- Bottom-3 class mAP50–95;
- Worst-class mAP50–95;
- full classwise AP50–95;
- `Biji Hitam Pecah` AP50–95;
- added CWCF parameter count.

Primary matched deltas:

```
V8N_CWCF1 - V8N_MATCHED
```

## Frozen screen decision

This first portability screen uses only two promotion criteria:

- Macro mAP50–95 delta > 0;
- Bottom-3 mAP50–95 delta > 0.

If both are positive:

`PASS_CONFIRMATION_SCREEN -> PAIRED_MULTI_SEED_CONFIRMATION`

If the aggregate metrics trade off:

`PARETO_TRADEOFF_SCREEN -> REVIEW_BEFORE_CONFIRMATION`

If there is no aggregate advantage:

`NO_ADVANTAGE_SCREEN -> RETAIN_V8N_MATCHED`

Worst-class AP and `Biji Hitam Pecah` AP are reported but are **not**
promotion gates. This prevents another target-specific validation chase after
the negative CWCFHNC1 result.

## Claim boundary

A passing seed-42 screen means only that the frozen CWCF1 mechanism is promising
on YOLOv8n under one matched seed. It does not establish final superiority,
multi-seed stability, or locked-test performance.

## Test lock

- no test image is extracted;
- no candidate or baseline evaluates the locked test;
- this screen cannot authorize final-test access.
