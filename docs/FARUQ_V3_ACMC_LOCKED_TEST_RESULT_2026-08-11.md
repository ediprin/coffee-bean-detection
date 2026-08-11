# Faruq-v3 ACMC Locked-Test Result — 2026-08-11

Status: final, `NOT_CONFIRMED`. Test is closed and further tuning is not
authorized.

## Protocol

- Models: optimization-matched D0FT versus ACMC1.
- Frozen seeds: 42, 123, and 2026.
- Test set: 129 independent Faruq parent images, 208 instances, all 21 classes.
- Identity audit: zero development parent/hash overlap and one selected image
  per test parent.
- Primary endpoint: paired three-seed macro mAP50-95 delta.
- Uncertainty: 1,000-iteration paired-parent bootstrap, seed 20260809.
- Bottom-three and worst-class AP are descriptive because rare-class test
  support is only 5--15 instances and 4--13 parents.

The original v1 eligibility gate remains `FAIL` because two classes did not
reach 10 instances and 5 parents. The support-qualified v2 amendment was
frozen before model inference and did not overwrite that result.

## Aggregate result

| Metric | D0FT mean ± SD | ACMC1 mean ± SD | Paired delta mean ± SD | Minimum delta | Improved seeds |
|---|---:|---:|---:|---:|---:|
| Macro mAP50-95 | 86.31% ± 0.95% | 87.55% ± 1.21% | +1.24% ± 1.14% | +0.35% | 3/3 |
| Bottom-3 mAP50-95 | 72.85% ± 3.70% | 76.13% ± 1.05% | +3.29% ± 4.20% | +0.07% | 3/3 |
| Worst-class mAP50-95 | 69.07% ± 2.43% | 73.08% ± 1.50% | +4.01% ± 3.84% | +0.14% | 3/3 |

Macro mAP50-95 gains by seed were +0.35, +0.84, and +2.53 percentage
points for seeds 42, 123, and 2026, respectively.

## Paired-parent bootstrap

- Custom paired point delta: +1.34 percentage points.
- 95% percentile interval: -0.41 to +3.09 percentage points.
- Probability of positive delta: 0.928.
- Frozen confirmation threshold: at least 0.950.

The standard Ultralytics point delta (+1.24 points) and custom bootstrap point
delta (+1.34 points) agree closely. Nevertheless, the interval includes zero
and the predeclared probability threshold was not reached.

## Decision

Two point-estimate criteria passed:

- positive mean macro gain;
- positive macro gain on at least two of three seeds (observed 3/3).

The paired-parent-bootstrap criterion failed (`0.928 < 0.950`). Therefore the
locked-test conclusion is **`NOT_CONFIRMED`**. The data support a consistently
positive and practically relevant trend, including improved lower-tail
metrics, but do not support a definitive superiority claim under the frozen
inferential gate.

## Technical interruptions

Before the final run, one attempt stopped before evaluation because the
locked-test YAML lacked an Ultralytics-required `train` schema key. A second
attempt completed D0FT seed 42 validation but was killed before writing a
report while starting a redundant prediction pass. Both were infrastructure
failures. The final runner used an evaluation-only schema alias and captured
bootstrap predictions from the same validation pass. No checkpoint, dataset,
threshold, seed, metric, or acceptance criterion changed, and no training was
performed.

Raw authoritative summary:
`Coffee_Bean_Detection/experiments/faruq-v3-acmc-locked-test-v2/faruq_v3_acmc_locked_test_summary.json`
in the shared Google Drive project.
