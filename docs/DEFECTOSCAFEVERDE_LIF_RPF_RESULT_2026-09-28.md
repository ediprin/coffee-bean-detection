# DefectosCafeVerde LIF-RPF Seed-42 Result

Date completed: 2026-09-28.

## Decision

**STOP_AFTER_SEED42. Retain `D0DIRECT`; do not run additional LIF-RPF seeds or
open test.**

The comparison used the frozen paper-backed
`defectoscafeverde-grouped-physical-v1` development split. Dataset-audit SHA,
official-pretrained SHA, native config, 50-epoch schedule, and seed 42 matched.
All 12 validation classes were present and test was not accessed.

| Model | Macro mAP50-95 | Bottom-3 | Worst class |
|---|---:|---:|---:|
| `D0DIRECT` | 91.62% | 86.73% | 84.52% |
| `LIFRPF1` | 90.82% | 85.72% | 84.04% |
| Delta | **-0.80 pp** | **-1.00 pp** | **-0.48 pp** |

Neither the overall route nor lower-tail route passed.

## Classwise result

Only three of twelve classes improved: `agrio` (+0.19 pp), `elefante`
(+0.35 pp), and `triangulo` (+0.36 pp). The largest losses were `broca`
(-2.63 pp), `oreja` (-2.62 pp), `partido` (-1.87 pp), and `normal`
(-1.25 pp). The intended illumination/reflectance cue therefore did not
produce a broad discrimination improvement under the controlled LED capture
domain.

## Validation-only gate attribution

The Drive checkpoint SHA256 was verified as
`6b093fa71917a1fce930d731a1e1af727edf3303c51c63c8ac7319480ae7a4ed`.
Its three raw scalar gates were `0.0046425`, `-0.0267029`, and `0.0939941`,
corresponding to bounded effective gains of only **+0.046%**, **-0.267%**, and
**+0.937%**.

The same checkpoint was evaluated on validation after setting all three gates
to zero in memory. No training or checkpoint rewrite was performed.

| Endpoint | Macro mAP50-95 | Bottom-3 | Worst class |
|---|---:|---:|---:|
| `LIFRPF1` active | 90.8214% | 85.7247% | 84.0359% |
| `LIFRPF1` zero gate | 90.8247% | 85.7098% | 84.0198% |
| Zero minus active | +0.0033 pp | -0.0149 pp | -0.0161 pp |

The zero-gate endpoint is practically identical to the active endpoint, yet it
remains below `D0DIRECT` by -0.795 pp Macro, -1.019 pp Bottom-3, and -0.501 pp
Worst. Attribution is therefore:

`TRAINING_PATH_DOMINANT_CUE_NEARLY_IGNORED`.

The LIF cue itself is not materially damaging inference. Instead, the detector
learned nearly zero use of the cue while the candidate training trajectory
settled on slightly worse native detector weights. This rejects the hypothesis
that conservative illumination preprocessing is a useful complementary signal
for this controlled-capture dataset.

## Claim boundary

This is a single-seed, validation-only mechanism screen plus a post-hoc
zero-gate attribution. It is not a locked-test claim and does not establish
performance under real illumination domain shift.

