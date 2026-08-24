# Faruq-v3 AF2_ORIENT Paired Confirmation

Status: **frozen before seed 123/2026 training**

## Question

Does the lower-tail improvement of unsigned 180-degree AF2 orientation folding
remain stable across seeds without sacrificing mean Macro mAP50-95?

## Frozen design

- Dataset: Faruq-v3 grouped development train/validation only.
- Test: locked and unavailable.
- Models: original AF2 control and `AF2_ORIENT` candidate.
- Seeds: 42, 123, and 2026.
- Seed 42 is reused from completed evidence; it is not retrained.
- Seeds 123/2026 start from their corresponding D0 checkpoints and use the
  same 50-epoch schedule as seed 42.
- AF2_ORIENT differs from AF2 only by folding signed 360-degree directions into
  unsigned 180-degree orientations while preserving one-degree bin resolution.
- No radial bands, combined arm, extra loss, test access, or hyperparameter
  selection is authorized.

## Paired gate

For every metric, delta means `AF2_ORIENT - AF2` at the same seed. PASS requires
all of:

1. mean Macro delta is not lower than zero;
2. Macro improves in at least two of three seeds;
3. mean Bottom-3 gain is at least +0.5 percentage point;
4. Bottom-3 improves in at least two of three seeds;
5. mean Worst-class delta is not lower than zero;
6. Worst-class improves in at least two of three seeds.

PASS retains AF2_ORIENT as the tail-strengthened AF2 variant. FAIL retains
original AF2. Neither outcome opens the locked test.
