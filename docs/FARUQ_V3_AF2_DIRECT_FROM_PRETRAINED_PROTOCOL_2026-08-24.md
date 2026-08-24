# Faruq-v3 AF2 direct-from-pretrained protocol — 2026-08-24

## Status

**FROZEN BEFORE TRAINING.** This protocol defines a validation-only seed-42 screen. The locked test split remains closed regardless of outcome.

## Research question

Does the retained AF2 input frontend improve fine-grained SNI-21 detection when it is active from the first optimization step starting from the official YOLO26n pretrained checkpoint, compared with a schedule- and initialization-matched native YOLO26n control?

This is intentionally different from the completed staged AF2 evidence, where AF2 was introduced after coffee-domain training/continuation.

## Arms

Both arms start from the **same exact official `yolo26n.pt` artifact** and the same 21-class target-head initialization.

Frozen pretrained artifact under Ultralytics 8.4.96:

```text
filename = yolo26n.pt
source release = ultralytics/assets v8.4.0
SHA-256 = 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
source classes = 80
```

| Arm | Detector | Input frontend | Initialization |
|---|---|---|---|
| `D0DIRECT` | YOLO26n P3–P5 | none | frozen official `yolo26n.pt` |
| `AF2DIRECT` | YOLO26n P3–P5 | retained AF2 | exact same frozen `yolo26n.pt` |

No D0, D0FT, AF2, AF2FS, or other coffee-trained checkpoint is accepted as an intermediate parent. The runner rejects any supplied pretrained file whose SHA-256 differs from the frozen artifact above; the 80-class source-head check is an additional guard.

## Matched target-head initialization

The official pretrained detector has an 80-class head whereas Faruq-v3 has 21 classes. Shape-incompatible target-head tensors therefore require fresh initialization. To prevent that randomness from becoming a hidden treatment difference, both arms construct their 21-class detector inside an isolated RNG fork with the same seed before loading the common pretrained source. A static preflight requires the complete persistent detector state of `D0DIRECT` and `AF2DIRECT` to be exactly equal after the common source transfer.

AF2 has no learned parameters, so after source transfer the only intended arm difference is the deterministic input operator used by `AF2DIRECT` during every tensor forward.

## Frozen AF2 operator

The experiment reuses `configs/afab/AF2_yolo26n_chaotic_amplitude.yaml` without tuning:

- mode: `af2`
- patch size: 32
- overlap: 0.50
- gamma: 0.10
- angular bins: 360
- chunk size: 128
- epsilon: `1e-8`
- channel processing: independent RGB channels
- overlapping reconstruction: fold/overlap averaging
- output gate: `x_AF2 = x + x * minmax(recover_AF2(x))`

The operator preserves tensor shape and does not crop, resize, translate, or warp object coordinates. This is a geometric property of the input transform, not a guarantee that learned box predictions remain numerically unchanged after training.

## Frozen training schedule

The two training blocks must be byte-for-value equivalent at runner preflight. Settings are inherited from the locked D0 and AF2 configs:

```text
seed          = 42
epochs        = 50
imgsz         = 640
batch         = 16
workers       = 2
patience      = 15
optimizer     = auto
pretrained    = true
cache         = false
close_mosaic  = 10
max_det       = 500
deterministic = true
```

Early stopping is valid under the shared `patience=15` rule; 50 epochs is the maximum, not a requirement that both arms stop at the same epoch.

## Dataset and test lock

Use the immutable Faruq-v3 grouped development archive only:

- train: 1,665 images / 2,986 annotations
- validation: 294 images / 526 annotations
- 21 SNI classes present in both development splits
- cross-split parent/exact-hash leakage gates already closed by the grouped dataset contract
- the development root must not contain a `test` directory

All model selection, screening, and mechanism diagnostics use validation only. The test split is not restored or opened.

## Static preflight gates

Training is authorized only if all of the following pass:

1. supplied `yolo26n.pt` SHA-256 exactly equals the frozen official artifact;
2. native and AF2 configs reference the same YOLO26n P3–P5 model YAML;
3. the two training schedules are exactly equal;
4. the AF2 mapping equals the frozen operator above;
5. the supplied pretrained source exposes the expected 80-class pretrained head;
6. `D0DIRECT` and `AF2DIRECT` have exactly the same persistent detector-state keys and tensors immediately after matched target-head initialization + source transfer;
7. detector parameter counts are equal;
8. AF2 contributes zero learned parameters;
9. an AF2 probe preserves BCHW shape, remains finite, and produces a nonzero input transformation;
10. the dataset audit passes and the test split remains absent.

The preflight records the pretrained SHA-256 and the common initialized detector-state fingerprint. The common state fingerprint is compared **within the same runtime** rather than frozen across PyTorch environments; the pretrained artifact SHA is the cross-runtime source lock. Resume is allowed only within a run directory whose frozen input contract is unchanged.

## Primary validation metrics

For each arm record:

- Macro mAP50–95
- Bottom-3 class mAP50–95
- Worst-class mAP50–95

The paired deltas are always:

`AF2DIRECT - D0DIRECT`.

## Mechanism diagnostic

After both best checkpoints are available, run the existing validation-only Faruq-v3 diagnostic with:

- image size 640
- match IoU 0.50
- confidence threshold 0.25
- NMS IoU 0.70
- `max_det=500`
- raw candidate counts 50/100/300/500

Headline mechanism quantities:

1. raw top-500 proposal accessibility — localization/proposal availability;
2. localization-conditioned Top-1 class accuracy — class discrimination after localization;
3. correct-decision recall.

The diagnostic is an attribution aid, not a substitute for the primary mAP/tail metrics.

## Prospective seed-42 screen decision

This is a **promotion screen**, not a final thesis confirmation. Thresholds are frozen before observing any AF2-direct result.

Localization safety for either route:

`Delta raw top-500 proposal accessibility >= -0.50 pp`.

### Route A — direct overall gain

All required:

- Macro delta >= +0.50 pp
- Bottom-3 delta >= -0.50 pp
- Worst delta >= -0.50 pp
- localization safety passes

### Route B — lower-tail Pareto signal

All required:

- Macro delta >= -0.20 pp
- Bottom-3 delta >= +1.00 pp
- Worst delta >= +1.00 pp
- localization safety passes

`PROMOTE_TO_3_SEED = Route A OR Route B`.

If the seed-42 screen passes, the exact direct protocol may be extended prospectively to seeds 123 and 2026 before any final direct-AF2 claim. If it fails, the direct-from-pretrained route is not promoted; the previously completed staged AF2 evidence remains the valid AF2 result.

## Claim boundary

A seed-42 PASS only means AF2-direct produced enough paired validation signal to justify a three-seed confirmation. It does **not** establish final superiority, untouched-test generalization, universal localization preservation, cross-dataset robustness, or deployment readiness.

A future three-seed confirmation would still require an explicit efficiency comparison because AF2 is parameter-free but not latency- or memory-free.
