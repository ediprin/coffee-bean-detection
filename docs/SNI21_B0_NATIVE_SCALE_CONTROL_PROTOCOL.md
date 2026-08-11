# SNI-21 B0 native-scale control

**Version:** 1.0

**Frozen:** 31 July 2026

**Status:** validation-derived diagnostic; no training and no test access

## Question

The original synthetic B0 arm matches the real-validation scene count
(`1--5` objects), but its objects are substantially smaller than those in
R0. This control asks whether that scale mismatch explains the B0 performance
collapse.

## Frozen comparison

Only one new arm is generated:

| Arm | Identities | Scenes | Objects | Prior | Visibility | Scale |
|---|---|---:|---:|---|---|---|
| B0 original | validation | 200 | 1--5 | empirical | mild | 2.5--5.5% |
| B0 native-scale | validation | 200 | 1--5 | empirical | mild | R0 q05--q95 |

The native scale is the q05--q95 interval of ground-truth box long side
divided by image long side in the already-completed R0 validation diagnosis.
It is derived programmatically; no test record is read.

The generator reuses the original validation object library, scene profile,
seed, class prior, visibility target, canvas, reuse limits, and placement
policy. Named RNG streams keep scene count, selected source assets, class draw,
and geometry target paired. The runner must verify the source-asset sequence
scene by scene and stop if pairing is not exact.

## Evaluation

Use the same frozen A0 YOLO26n checkpoint and the same settings as the completed
density evaluation:

- image size 640;
- confidence 0.001 for AP/diagnostic candidate collection;
- NMS IoU 0.7;
- diagnostic IoU 0.5;
- `max_det=300`.

Report mAP50--95, mAP50, precision, recall, macro/bottom-three/worst AP,
proposal recall, conditional class accuracy, wrong-class rate, miss rate, and
saturation. Count bias at confidence 0.001 remains diagnostic, not an
operational counting claim.

## Interpretation rule

For mAP50--95:

```text
recovery = (B0_native - B0_original) / (R0 - B0_original)
```

- recovery >= 0.50: scale explains the majority of the observed B0 gap;
- 0.20 <= recovery < 0.50: scale is a material but partial cause;
- recovery < 0.20: scale alone does not explain the collapse.

This rule attributes the diagnostic gap only. It does not validate synthetic
realism or authorize model training.

## Boundaries

- No detector training is run.
- Test images and test annotations remain untouched.
- The arm is development-correlated because it uses validation identities and
  R0 validation geometry.
- B1--B3 are not regenerated.
- No model-selection or real-dense claim is allowed from this control.
