# Faruq-v3 AF2 × D-FINE-N cross-architecture transfer protocol — 2026-08-29

## Status

**FROZEN BEFORE TRAINING.** This is a validation-only seed-42 transfer screen. The Faruq-v3 locked test split remains closed regardless of outcome. This experiment does not replace the completed AF2 evidence on YOLO26 and does not yet modify the formal thesis proposal.

## Research question

Does the retained parameter-free AF2 input frontend improve fine-grained SNI-21 detection when transferred without retuning to a modern lightweight DETR-family detector, D-FINE-N, relative to an initialization-, schedule-, dataset-, and evaluation-matched native D-FINE-N control?

The scientific comparison is **within D-FINE**:

`AF2 + D-FINE-N - D-FINE-N`

It is not a claim that D-FINE-N is superior to YOLO26n or vice versa.

## Why D-FINE-N

The official D-FINE repository reports D-FINE-N at approximately 4M parameters, 7 GFLOPs, and 42.8 COCO AP, making it substantially closer in capacity to YOLO26n than RT-DETRv3-R18 while still representing a different detector family (DETR-style end-to-end detection).

Frozen upstream source:

```text
repository = Peterande/D-FINE
commit     = 956d1709314c2c6a4df6f34de232054578a7449f
model      = D-FINE-N / HGNetv2-B0
base cfg   = configs/dfine/custom/dfine_hgnetv2_n_custom.yml
pretrain   = official dfine_n_coco.pth from release dfinev1.0
```

The release asset itself must be SHA-256 fingerprinted in the execution runtime before training, and both arms must use the exact same downloaded file.

## Arms

| Arm | Detector | Input frontend | Initialization |
|---|---|---|---|
| `DFN0` | D-FINE-N | none | exact same official D-FINE-N COCO checkpoint |
| `DFN_AF2` | D-FINE-N | retained AF2 | exact same official D-FINE-N COCO checkpoint |

The only intended treatment difference is the deterministic AF2 input operator.

## Frozen AF2 operator

Reuse the retained AF2 implementation already frozen in this repository:

```text
mode         = af2
patch_size   = 32
overlap      = 0.50
gamma        = 0.10
angular_bins = 360
chunk_size   = 128
eps          = 1e-8
RGB channels = independent
overlap reconstruction = fold / overlap averaging
output       = x + x * minmax(recover_AF2(x))
```

No AF2 parameter is tuned for D-FINE-N.

### Placement

D-FINE's official dataloader converts the resized image to float32 and scales it to approximately `[0,1]`. AF2 is inserted at the beginning of `DFINE.forward()` before the HGNetv2 backbone:

```text
native: batch tensor -> HGNetv2 -> HybridEncoder -> DFINETransformer
AF2:    batch tensor -> frozen AF2 -> HGNetv2 -> HybridEncoder -> DFINETransformer
```

This keeps AF2 on the batched model path rather than running FFT preprocessing inside CPU dataloader workers. AF2 preserves BCHW shape and does not alter box coordinates.

The retained residual operator is not clipped after AF2; for a base input in `[0,1]`, the theoretical AF2 output may reach `[0,2]`, consistent with the retained YOLO AF2 implementation.

## Dataset

Use the same immutable Faruq-v3 grouped development data used by the direct AF2 screen:

```text
train      = 1,665 images / 2,986 annotations
validation = 294 images / 526 annotations
classes    = 21
```

The grouped train/validation split must remain identical. The test directory must be absent from the execution package.

D-FINE requires COCO-format annotations. Conversion from the frozen YOLO-format development labels is a format conversion only:

- image bytes are not modified;
- train/validation membership is not changed;
- class ordering is preserved;
- normalized YOLO boxes are converted to absolute COCO `xywh` boxes;
- no test data are created or read.

The generated COCO manifests must be fingerprinted and reused by both arms.

## Training schedule

For this cross-architecture experiment, fairness is defined within D-FINE, not by forcing YOLO's training recipe onto D-FINE.

Both arms therefore inherit the official D-FINE-N custom-dataset recipe at the frozen upstream commit, including its optimizer and 220-epoch maximum schedule, and are fine-tuned from the same official `dfine_n_coco.pth` checkpoint.

Frozen screen seed:

```text
seed = 42
AMP  = enabled for both arms
```

No training hyperparameter may differ between `DFN0` and `DFN_AF2`. No checkpoint from a coffee-trained YOLO or D-FINE run may be used as parent initialization.

## Static preflight requirements

Training is authorized only if all of the following hold:

1. upstream D-FINE checkout is exactly commit `956d1709314c2c6a4df6f34de232054578a7449f`;
2. both arms use D-FINE-N / HGNetv2-B0 and the same 21-class target configuration;
3. both arms use the exact same `dfine_n_coco.pth` bytes and recorded SHA-256;
4. both arms use the same train/validation COCO manifests and recorded SHA-256 values;
5. AF2 mapping exactly matches the frozen operator above;
6. AF2 contributes zero learned parameters;
7. a deterministic AF2 probe preserves BCHW shape, remains finite, and changes the input non-trivially;
8. native and AF2 detector parameter counts are exactly equal;
9. before optimization, persistent detector state keys and tensors are equal after common target-model construction and common pretrained transfer; AF2 non-persistent geometry buffers must not enter the detector-state fingerprint;
10. the locked test split is absent.

## Primary validation metrics

Record for each arm:

- mAP50-95 / COCO-style AP as the primary detector metric;
- per-class AP50-95;
- Macro class AP50-95;
- Bottom-3 class AP50-95;
- Worst-class AP50-95.

All paired deltas are:

`DFN_AF2 - DFN0`.

The same confidence/postprocessing and evaluator settings must be used for both arms.

## Prospective seed-42 screen decision

This is a promotion screen, not final confirmation. To keep the decision comparable with the existing AF2-direct screen, the same validation-scale tail thresholds are reused where the metric definitions are identical.

### Route A — overall gain

All required:

- Macro AP50-95 delta >= +0.50 percentage point;
- Bottom-3 delta >= -0.50 pp;
- Worst delta >= -0.50 pp.

### Route B — lower-tail Pareto signal

All required:

- Macro delta >= -0.20 pp;
- Bottom-3 delta >= +1.00 pp;
- Worst delta >= +1.00 pp.

`PROMOTE_TO_3_SEED = Route A OR Route B`.

A localization-specific safety gate is intentionally not frozen yet for D-FINE because the existing YOLO raw-proposal diagnostic is architecture-specific. No YOLO proposal diagnostic will be naively reused for DETR queries. If a D-FINE-compatible localization diagnostic is later added, it must be prospectively defined before its result is used for a decision.

## If seed 42 passes

Extend the exact same D-FINE protocol prospectively to seeds:

```text
42, 123, 2026
```

No AF2 retuning is allowed. The locked test remains closed until a separate decision explicitly authorizes final evaluation.

## Claim boundary

A seed-42 PASS would support only:

> AF2 shows enough validation signal on D-FINE-N to justify a multi-seed cross-architecture confirmation.

It would not establish:

- universal transfer across detector families;
- D-FINE superiority over YOLO26;
- untouched-test confirmation;
- deployment robustness;
- improved localization mechanism;
- computational efficiency.

If the transfer fails, the valid interpretation is that the retained AF2 effect did not transfer under this frozen D-FINE-N protocol. That outcome does not invalidate the completed AF2 evidence on YOLO26.
