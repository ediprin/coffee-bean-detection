# Faruq-v3 Top-Control Paired Three-Seed Confirmation Protocol

Date frozen: 2026-08-21

Evaluation: grouped Faruq-v3 development validation only

Test: locked and unavailable to every runner

## Question

Several seed-42 optimization controls produced Macro, Bottom-3, and Worst-class
mAP50-95 above 80%. This protocol tests whether those observations persist
under seed-matched continuation rather than promoting a single favorable seed.

## Frozen arms

| Arm | Seed-matched source | Role |
|---|---|---|
| `FCT0` | `STB1` | STB continuation/optimization control |
| `AF2R0` | `AF2` | equal-parameter zero-information continuation control |
| `AF2R1` | `AF2` | illumination-conditioned residual gate |
| `AF2CAL3` | `AF2` | three-scalar RGB residual calibration |

Seed 42 is reused from frozen evidence. Only seeds 123 and 2026 are newly
trained. `AF2FT30` is not retrained: its exact frozen schedule has already been
confirmed as `AF2CT30` for seeds 42/123/2026.

## Comparisons

- `FCT0 - STB1`
- `AF2R0 - AF2`
- `AF2R1 - AF2R0`
- `AF2CAL3 - AF2CT30`

Every comparison uses the same seed on both sides. No comparison is made to a
seed-42 checkpoint when evaluating seed 123 or 2026.

## Frozen gate

Each arm is decided independently and passes only if all conditions hold:

1. mean Macro gain is at least +0.5 percentage point;
2. Macro improves in at least two of three seeds;
3. mean Bottom-3 is not lower;
4. Bottom-3 improves in at least two of three seeds;
5. mean Worst-class decline is no greater than one percentage point.

The protocol does not select hyperparameters, open the locked test, or authorize
additional tuning. A passing control may be retained as an optimization
protocol; it is not automatically an architectural contribution.

## Execution contract

- Allowed new seeds: exactly 123 and 2026.
- Source checkpoints must record and match the requested seed.
- Dataset root must not contain a `test` directory.
- Static audit is rerun against each seed-matched AF2 checkpoint for AF2R and
  AF2CAL arms.
- Training writes `last.pt` and `best.pt` directly under the shared Drive
  project and uses an exclusive per-arm/per-seed lock.
- A completed result must report all 21 validation classes and
  `test_images_accessed: false`.
