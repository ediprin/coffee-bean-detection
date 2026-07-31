# SNI-21 Paired Full-Frame Context Control

## Question

R0 repaste retained 91.83% of mAP50-95 and 88.81% of conditional class
accuracy. Cutout extraction is therefore not the primary explanation for the
B0 collapse. This no-training control isolates the next factor: procedural
background without cropping or rescaling the source image.

## Three paired arms

Only real validation images containing exactly one labeled object are eligible.
The original canvas dimensions and target bbox are preserved in every arm:

1. `FC0_original_fullframe`: untouched source image;
2. `FC1_repaste_real_fullframe`: extracted cutout pasted at its original bbox on
   the untouched source image;
3. `FC2_repaste_procedural_fullframe`: the exact FC1 cutout and bbox pasted on a
   full-frame procedural background.

FC0 versus FC1 measures cutout/repaste damage. FC1 versus FC2 changes the
background while keeping foreground pixels, geometry, scale, resolution, and
canvas size fixed. All three arms are encoded with the same JPEG quality and
chroma-subsampling settings so encoding is not arm-specific.

The superseded local-crop design is invalid for attribution because resizing a
small context window to the detector input caused a new object-scale and
context distribution shift. Its output must not be interpreted as evidence
about procedural background.

## Selection

- Validation identities only; test is never opened.
- The source image must contain exactly one labeled object.
- Source dataset, parent identity, and class must match the object library.
- Up to 20 deterministic samples per class are used.
- At least 150 samples and 15 represented classes are required before
  inference.

The implementation preflight on the frozen validation archive found 172
eligible images across 18 classes, satisfying the gate without test access.

## Frozen model and metrics

The A0 seed-42 checkpoint, SHA-256, inference size, confidence, NMS IoU, and
diagnostic IoU are fixed. Report mAP50-95, proposal recall, and conditional
class accuracy for every arm.

Background attribution is valid only when both safeguards pass:

1. FC0 retains at least 50% of R0 mAP50-95 and conditional class accuracy. This
   broad safeguard permits the deliberately different single-object subset but
   rejects another catastrophic distribution shift.
2. FC1 retains at least 80% of FC0 mAP50-95 and conditional class accuracy.

If either safeguard fails, the background result is `inconclusive_control_invalid`.
Otherwise the lower FC2/FC1 retention determines the diagnosis:

- at least 80%: procedural background is not the primary cause;
- 50% to below 80%: procedural background is a material partial cause;
- below 50%: procedural background is a dominant cause.

These are frozen diagnostic thresholds, not significance tests.

## Boundaries

- No training.
- Test remains locked.
- This control does not test object-level photometric transforms, shadows,
  overlap, or dense placement.
- Results are development diagnostics and cannot establish real dense
  performance.
