# Faruq-v3 WAV1 Mechanism Factorization — Stage-1 Seed-42 Result

Date recorded: 2026-08-19  
Branch: `agent/wav1-mechanism-factorization`  
Scope: Faruq-v3 grouped development validation only  
Seed: **42**  
Locked test: **closed / not accessed**  
Protocol basis: `docs/FARUQ_V3_WAV1_MECHANISM_FACTORIZATION_PROTOCOL.md`

## Status

Stage-1 all-in-one Kaggle screening completed for all four frozen causal arms:

- `HP1`
- `WAV_L1`
- `WAV_L2`
- `WAV_RAWFUSE`

The notebook reported `MECHANISTIC_REVIEW_REQUIRED`, as intended by the frozen protocol. This document records the completed seed-42 result and the mechanistic review. It does **not** authorize seed 123/2026 or locked-test evaluation.

## Frozen references

| Arm | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 |
|---|---:|---:|---:|
| `D0FT` | 0.866887 | 0.749809 | 0.720224 |
| `WAV1` | 0.884105 | 0.832761 | 0.820349 |

The corresponding seed-42 WAV1 gains versus D0FT are:

- Macro: **+1.7218 pp**
- Bottom-3: **+8.2952 pp**
- Worst-class: **+10.0125 pp**

## Stage-1 results

| Arm | Macro | Bottom-3 | Worst | Δ Macro vs D0FT | Δ Bottom-3 vs D0FT | Δ Worst vs D0FT |
|---|---:|---:|---:|---:|---:|---:|
| `HP1` | 0.878555 | 0.790939 | 0.756741 | +1.1668 pp | +4.1130 pp | +3.6517 pp |
| `WAV_L1` | **0.885721** | **0.839933** | **0.820947** | **+1.8833 pp** | **+9.0125 pp** | **+10.0723 pp** |
| `WAV_L2` | 0.872437 | 0.804273 | 0.764212 | +0.5550 pp | +5.4464 pp | +4.3988 pp |
| `WAV_RAWFUSE` | 0.879848 | 0.823032 | 0.813290 | +1.2961 pp | +7.3224 pp | +9.3066 pp |

## WAV1 gain preservation

The protocol defines descriptive gain preservation for headline metric `T` as:

\[
R_T(M)=\frac{T(M)-T(D0FT)}{T(WAV1)-T(D0FT)}.
\]

| Arm | Macro preservation | Bottom-3 preservation | Worst preservation |
|---|---:|---:|---:|
| `HP1` | 0.6777 | 0.4958 | 0.3647 |
| `WAV_L1` | **1.0938** | **1.0865** | **1.0060** |
| `WAV_L2` | 0.3223 | 0.6566 | 0.4393 |
| `WAV_RAWFUSE` | 0.7527 | 0.8827 | 0.9295 |

## Direct seed-42 contrast versus WAV1

`WAV_L1` is numerically above the frozen WAV1 seed-42 reference by:

- Macro: **+0.1615 pp**
- Bottom-3: **+0.7173 pp**
- Worst-class: **+0.0599 pp**

These are single-seed differences and must **not** be interpreted as general superiority over WAV1.

## Mechanistic review

### 1. Generic local-detail enhancement is helpful but insufficient to explain WAV1

`HP1` improves all three headline metrics versus D0FT, so a generic fixed local high-pass cue has a positive seed-42 effect. However, it preserves only about 49.6% of the WAV1 Bottom-3 gain and 36.5% of the Worst-class gain. Under this tested control, the WAV1 effect is therefore not explained by generic high-pass enhancement alone.

Claim boundary: this only compares against the frozen `HP1` control. It does not establish that Haar wavelets outperform every possible high-pass or local-detail operator.

### 2. Level-1 Haar detail is the dominant seed-42 mechanistic candidate

`WAV_L1` preserves approximately 109.4% of the WAV1 Macro gain, 108.6% of the Bottom-3 gain, and 100.6% of the Worst-class gain. At seed 42, level-1 Haar detail alone therefore reproduces essentially the entire observed WAV1 lower-tail benefit.

This is the strongest Stage-1 evidence and makes `WAV_L1` the mechanistic survivor for a separate confirmation protocol.

### 3. Two-level / multiscale necessity is not supported by this seed

`WAV_L2` remains beneficial versus D0FT but is substantially weaker than `WAV_L1`, especially on the lower-tail metrics. Since `WAV_L1` alone matches or slightly exceeds the frozen two-level WAV1 reference at seed 42, the current result does not support a claim that the second Haar level is necessary for the WAV1 effect.

### 4. Per-level equalization appears secondary rather than fundamental

`WAV_RAWFUSE` retains 88.3% of the WAV1 Bottom-3 gain and 92.9% of the Worst-class gain even after removing WAV1's per-level min-max equalization before fusion. This indicates that per-level equalization is not the sole or primary explanation for the effect, although WAV1 remains modestly above RAWFUSE on the three headline metrics.

## Stage-1 decision

Mechanistic survivor:

\[
\boxed{\texttt{WAV\_L1}}
\]

Current interpretation, restricted to seed 42:

> The strongest supported explanation for the confirmed WAV1 effect is a fine-scale level-1 Haar detail-energy spatial gate. Generic local-detail enhancement contributes partially; level-2 detail contributes partially; two-level fusion and per-level scale equalization are not required to reproduce the full seed-42 effect.

This is a **mechanistic screening conclusion**, not a final cross-seed or test-set claim.

## Next-step boundary

A new protocol must be frozen before any additional training. The intended next question is:

> Is single-level `WAV_L1` validation-robust versus seed-matched D0FT across seeds 42/123/2026?

This Stage-1 result does **not** by itself authorize:

- seed 123 or seed 2026 training;
- post-result tuning of `WAV_L1`;
- orientation-specific `LH`/`HL`/`HH` factorization;
- input-vs-classification placement experiments;
- Faruq locked-test access.

## Artifact note

The completed Kaggle notebook reported:

- final snapshot: `/kaggle/working/wav1-factorization-stage1-output.zip`
- generated report: `/kaggle/working/wav1-factorization-stage1-v1/val_reports/wav1_factorization_seed42_report.json`

The simplified notebook did not require the full frozen WAV1 per-class JSON, so per-class delta correlation against WAV1 is not part of this Stage-1 record.
