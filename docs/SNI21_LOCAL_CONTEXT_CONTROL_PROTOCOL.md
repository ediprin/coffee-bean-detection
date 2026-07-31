# SNI-21 Paired Local-Context Control

## Question

R0 repaste retained 91.83% of mAP50-95 and 88.81% of conditional class
accuracy. Cutout extraction is therefore not the primary explanation for the
B0 collapse. This no-training control isolates the next factor: procedural
background.

## Three paired arms

The same isolated validation object, class, bbox, cutout, and local position are
used in three arms:

1. `LC0_original_context`: original RGB patch;
2. `LC1_repaste_real_context`: extracted cutout pasted at its original bbox on
   the same RGB patch;
3. `LC2_repaste_procedural_context`: the exact LC1 cutout and bbox pasted on the
   procedural background used by the density generator.

LC0 versus LC1 measures local cutout/repaste damage. LC1 versus LC2 changes the
background while keeping foreground pixels and geometry fixed.

## Selection

- Validation identities only; test is never opened.
- Source dataset, parent identity, and class must match.
- Repeated same-class objects are matched by sorted log aspect ratio.
- The context window is three times the object long side.
- Edge objects and windows intersecting another labeled object are excluded.
- Up to 20 deterministic samples per class are used.
- At least 150 samples and 15 represented classes are required before
  inference.

## Frozen model and metrics

The A0 seed-42 checkpoint, SHA-256, inference size, confidence, NMS IoU, and
diagnostic IoU are fixed. Report mAP50-95, proposal recall, and conditional
class accuracy for every arm.

For LC1 to LC2, compute both metric retentions. The lower retention determines
the diagnostic interpretation:

- at least 80%: procedural background is not the primary cause;
- 50% to below 80%: procedural background is a material partial cause;
- below 50%: procedural background is a dominant cause.

These are frozen diagnostic thresholds, not significance tests.

## Boundaries

- No training.
- Test remains locked.
- This control does not test rotation, object-level photometric transforms,
  shadows, overlap, or dense placement.
- Results are development diagnostics and cannot establish real dense
  performance.
