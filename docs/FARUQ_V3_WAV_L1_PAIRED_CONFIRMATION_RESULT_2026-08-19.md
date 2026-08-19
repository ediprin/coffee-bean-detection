# Faruq-v3 WAV_L1 Paired Multiseed Confirmation Result

Date: 2026-08-19  
Branch: `agent/wav1-mechanism-factorization`  
Protocol: `docs/FARUQ_V3_WAV_L1_PAIRED_CONFIRMATION_PROTOCOL_2026-08-19.md`  
Evaluation: Faruq-v3 grouped development validation only  
Locked test accessed: **no**

## Result status

The frozen single-level Haar detail gate `WAV_L1` completed the planned confirmation seeds 123 and 2026. Seed 42 is reused from the frozen Stage-1 mechanism-factorization evidence and was not retrained.

Under the pre-frozen five-criterion gate versus seed-matched D0FT, the observed three-seed result is **PASS**.

This record is based on the completed validation outputs for seeds 42/123/2026. It is not a locked-test result and is not a superiority claim against two-level WAV1 or other candidate methods.

## Frozen WAV_L1 implementation

`WAV_L1` is the Stage-1 single-level factorization arm:

1. RGB -> Rec.709 luminance.
2. One orthonormal Haar DWT.
3. Level-1 detail energy:

\[
D_1=\sqrt{LH_1^2+HL_1^2+HH_1^2+\epsilon}.
\]

4. Bilinear resize to input resolution.
5. Frozen `stable_minmax_spatial` normalization.
6. Same RGB-replicated residual spatial gate:

\[
x'=x+x\odot N(D_1).
\]

The frontend remains parameter-free and state-free. The native YOLO26n-P3 detector is otherwise unchanged.

## Per-seed WAV_L1 validation results

| Seed | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 |
|---:|---:|---:|---:|
| 42 | 88.5721% | 83.9933% | 82.0947% |
| 123 | 87.1287% | 77.4281% | 73.8865% |
| 2026 | 86.3213% | 77.8208% | 74.4834% |

The exact candidate values recorded by the completed runs are:

- seed 42: Macro `0.885720537714217`, Bottom-3 `0.8399334705085897`, Worst `0.8209474694929713`;
- seed 123: Macro `0.8712866142181733`, Bottom-3 `0.7742813016388482`, Worst `0.738865224518935`;
- seed 2026: Macro `0.8632134001510495`, Bottom-3 `0.7782082231737212`, Worst `0.7448335483527468`.

## Seed-matched D0FT context

The frozen D0FT references used by the paired-confirmation family are:

| Seed | Macro | Bottom-3 | Worst |
|---:|---:|---:|---:|
| 42 | 86.6887% | 74.9809% | 72.0224% |
| 123 | 86.3509% | 74.7514% | 67.8296% |
| 2026 | 86.8096% | 79.9933% | 79.3107% |

Using these frozen displayed references, the paired WAV_L1 minus D0FT deltas are:

| Seed | Delta Macro | Delta Bottom-3 | Delta Worst |
|---:|---:|---:|---:|
| 42 | +1.8833 pp | +9.0125 pp | +10.0723 pp |
| 123 | +0.7778 pp | +2.6767 pp | +6.0569 pp |
| 2026 | -0.4883 pp | -2.1725 pp | -4.8273 pp |

Seed 2026 is negative on all three headline metrics and is retained without modification. Therefore `WAV_L1` must not be described as winning every seed or as seed-insensitive.

## Three-seed aggregate

Observed WAV_L1 aggregate:

| Metric | D0FT mean | WAV_L1 mean ± sample SD | Mean paired delta |
|---|---:|---:|---:|
| Macro | 86.6164% | **87.3407 ± 1.1402%** | **+0.7243 pp** |
| Bottom-3 | 76.5752% | **79.7474 ± 3.6823%** | **+3.1722 pp** |
| Worst | 73.0542% | **76.8215 ± 4.5765%** | **+3.7673 pp** |

Minor last-decimal differences may appear when the machine decision script reads the full-precision D0FT JSON rather than the rounded values displayed in the historical result document. The gate outcome is not affected by this display rounding.

## Frozen gate decision

The confirmation protocol requires all five conditions:

| Criterion | Observed result | Decision |
|---|---:|---|
| Mean Macro gain >= +0.5 pp | about +0.724 pp | **PASS** |
| Macro improves in at least 2/3 seeds | seeds 42 and 123 | **PASS** |
| Mean Bottom-3 not lower | about +3.172 pp | **PASS** |
| Bottom-3 improves in at least 2/3 seeds | seeds 42 and 123 | **PASS** |
| Mean Worst decline no greater than 1 pp | mean is about +3.767 pp | **PASS** |

Final protocol-level conclusion:

\[
\boxed{\texttt{WAV\_L1: PASS vs seed-matched D0FT across seeds 42/123/2026}}
\]

The valid wording is:

> The frozen single-level Haar level-1 detail-energy spatial gate is validation-robust versus seed-matched D0FT across the three planned seeds, with positive aggregate Macro and lower-tail gains but a negative seed-2026 direction.

## Descriptive comparison with confirmed two-level WAV1

Previously confirmed two-level WAV1 aggregate:

- Macro: 87.4010%
- Bottom-3: 79.7892%
- Worst: 77.0131%

Observed single-level WAV_L1 aggregate:

- Macro: 87.3407%
- Bottom-3: 79.7474%
- Worst: 76.8215%

Descriptive WAV_L1 minus WAV1 differences are therefore only approximately:

- Macro: -0.0603 pp
- Bottom-3: -0.0418 pp
- Worst: -0.1916 pp

Relative to the D0FT aggregate gain, WAV_L1 descriptively retains approximately:

- 92% of WAV1 Macro gain;
- 99% of WAV1 Bottom-3 gain;
- 95% of WAV1 Worst-class gain.

This is **not** a formal equivalence or non-inferiority test. It does, however, support the mechanistic interpretation that the level-1 fine-scale Haar detail term explains most of the confirmed two-level WAV1 effect under the current three-seed validation evidence.

## Interpretation boundary

Supported:

- `WAV_L1` is validation-robust versus D0FT in aggregate under the frozen three-seed protocol.
- Lower-tail gains remain substantially larger than the mean Macro gain.
- Level-1 detail is the strongest retained mechanism from the seed-42 factorization and preserves nearly all of the two-level WAV1 aggregate benefit descriptively.
- Seed sensitivity remains: seed 2026 is negative.

Not supported:

- general superiority of WAV_L1 over two-level WAV1;
- general superiority over all high-pass/detail operators;
- universal seed stability;
- locked-test generalization;
- a claim that the effect is specifically classification-only rather than shared representation/visibility.

## Next-step boundary

Do not retune WAV_L1, add a fourth seed, reopen level-2 fusion, or access the locked test under this confirmation result.

Any next experiment must have a newly frozen protocol. The most defensible next questions are error decomposition / matched localization-vs-classification diagnosis and, only after that, placement analysis.
