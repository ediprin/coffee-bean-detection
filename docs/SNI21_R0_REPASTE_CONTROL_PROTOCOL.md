# SNI-21 R0 Repaste Control Protocol

## Question

The density benchmark collapsed even at B0. Native object scale recovered only
about six percent of the R0-to-B0 mAP gap. This control asks a narrower question:

> Does extracting and alpha-compositing validation cutouts already destroy the
> information used by the frozen A0 detector, before procedural background,
> density, and new placement are introduced?

This is a development diagnostic. It is not training augmentation and it is not
a final test.

## Frozen comparison

The same A0 checkpoint and inference configuration are used for:

1. `R0_real_val`: existing real validation result;
2. `R0_real_val_repaste`: validation cutouts pasted back at the corresponding
   class, parent identity, position, and native bounding-box size on their real
   validation scene.

The repaste image is written as PNG. Pixels outside pasted boxes therefore equal
the decoded R0 image and are not changed by JPEG recompression.

## Matching and provenance

The persisted A0 archive does not retain raw COCO annotation IDs. Pairing is
therefore restricted to:

```text
source_dataset + source_parent_id + class_id
```

Within repeated instances of one class, boxes and assets are paired by sorted
log aspect ratio. No cross-class assignment is allowed. Coverage and per-class
coverage are written before inference. Evaluation is forbidden when total
coverage is below 85 percent.

Only `source_split=val` assets may be loaded. Test images and train identities
must not be opened. The existing checkpoint SHA-256 must match the completed
density evaluation.

## Metrics and interpretation

Report mAP50-95, proposal recall at IoU 0.5, and conditional class accuracy. Two
retentions are computed against R0:

```text
mAP retention = mAP(repasted) / mAP(R0)
class retention = conditional accuracy(repasted) / conditional accuracy(R0)
```

The lower retention freezes the diagnostic interpretation:

- at least 80%: repaste is not the primary cause;
- 50% to below 80%: repaste is a material partial cause;
- below 50%: repaste is a dominant cause.

These thresholds are diagnostic gates, not statistical significance claims.

## Boundaries

- No training is executed.
- Test remains locked.
- This control does not validate procedural backgrounds or dense realism.
- The aspect-ratio match is audited approximation, not recovered COCO annotation
  identity. Coverage and mismatch examples must accompany the result.
