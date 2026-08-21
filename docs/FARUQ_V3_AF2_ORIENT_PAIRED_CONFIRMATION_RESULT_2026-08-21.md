# Faruq-v3 AF2_ORIENT Paired Confirmation Result

Date: 2026-08-21

Protocol: `docs/FARUQ_V3_AF2_ORIENT_PAIRED_CONFIRMATION_PROTOCOL_2026-08-21.md`

Evaluation: grouped Faruq-v3 validation, paired seeds 42/123/2026

Test accessed: **no**

## Result

Each `AF2_ORIENT` run started from the seed-matched D0 checkpoint and used the
same 50-epoch schedule as its completed original-AF2 control. Seed 42 was
reused from the frozen isolated screening rather than retrained.

| Seed | Model | Macro mAP50-95 | Bottom-3 | Worst class |
|---:|---|---:|---:|---:|
| 42 | AF2 | 88.20% | 80.04% | 79.35% |
| 42 | AF2_ORIENT | **88.33%** | **81.38%** | **80.14%** |
| 123 | AF2 | 88.22% | 78.22% | 76.46% |
| 123 | AF2_ORIENT | **89.22%** | **79.06%** | **77.94%** |
| 2026 | AF2 | **87.40%** | **79.85%** | **78.65%** |
| 2026 | AF2_ORIENT | 87.23% | 77.32% | 71.90% |

Paired deltas in percentage points (`AF2_ORIENT - AF2`):

| Metric | Seed 42 | Seed 123 | Seed 2026 | Mean delta | Improved seeds |
|---|---:|---:|---:|---:|---:|
| Macro | +0.13 | +1.00 | -0.17 | **+0.32** | 2/3 |
| Bottom-3 | +1.34 | +0.84 | -2.53 | **-0.12** | 2/3 |
| Worst class | +0.79 | +1.48 | -6.75 | **-1.50** | 2/3 |

The three-seed means were 87.94% versus 88.26% Macro, 79.37% versus
79.25% Bottom-3, and 78.15% versus 76.66% Worst-class for AF2 versus
AF2_ORIENT, respectively.

## Frozen-gate decision

**FAIL -- retain original AF2.**

AF2_ORIENT passed the two Macro criteria and improved all three metrics in two
of three seeds. It nevertheless failed the frozen lower-tail requirements:

- mean Bottom-3 gain was -0.12 point instead of at least +0.5 point;
- mean Worst-class delta was -1.50 points instead of non-negative.

The positive mean Macro result therefore does not justify calling the variant
tail-strengthened or seed-stable. No additional orientation tuning, combined
radial-orientation arm, or locked-test evaluation is authorized by this result.

## Seed-2026 error attribution

The seed-2026 failure was not a uniform detector collapse. Orientation folding
redistributed AP between classes while leaving Macro nearly unchanged. The
largest losses versus original AF2 were:

| Class | AF2 | AF2_ORIENT | Delta |
|---|---:|---:|---:|
| `biji_muda` | 84.72% | 71.90% | **-12.82** |
| `biji_berlubang_lebih_satu` | 93.42% | 86.13% | -7.28 |
| `tanah_batu_ranting_besar` | 90.30% | 85.19% | -5.11 |
| `biji_berkulit_tanduk` | 87.32% | 82.72% | -4.60 |
| `kopi_gelondong` | 95.08% | 92.02% | -3.06 |

At the same time, `biji_normal` (+6.86), `biji_hitam_sebagian` (+6.06),
`biji_hitam_pecah` (+5.94), and `biji_bertutul_tutul` (+3.86 points) improved.
This opposing movement explains why the Macro delta was only -0.17 point while
the Worst-class metric fell sharply.

## Mechanistic interpretation

AF2_ORIENT folds antipodal FFT directions (`theta` and `theta + 180 degrees`)
into the same angular-density bin. For real-valued image patches, Fourier
magnitudes are already approximately conjugate-symmetric, so this folding adds
little independent information. It still changes the entropy-conditioned hard
threshold and therefore the recovered residual cue. During full detector
training, that small nonlinear input change can lead to different class
trade-offs across seed-matched optimization trajectories.

The evidence supports a bounded conclusion: unsigned orientation folding is a
useful one-seed perturbation and improves Macro in two of three seeds, but it
does not reliably strengthen the difficult-class tail. This is a negative
factorization result, not evidence of an implementation failure or a general
failure of original AF2.

Machine-readable evidence:
`docs/evidence/FARUQ_V3_AF2_ORIENT_PAIRED_CONFIRMATION_2026-08-21.json`.
