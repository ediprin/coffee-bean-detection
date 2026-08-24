# SNI-21 VA-DCP pilot result

**Recorded:** 30 July 2026
**Protocol:** `SNI21_VADCP_SCREENING_PROTOCOL.md`
**Evaluation:** real sparse validation only; locked test was not opened

## Frozen screening setup

- detector: YOLO26n;
- input: 640 px;
- epochs: 10;
- seed: 42;
- classes: 21;
- A0: real sparse train;
- A1: A0 plus 200 ordinary dense copy-paste scenes;
- A2: A0 plus 200 visibility-aware dense copy-paste scenes.

The synthetic arms were paired on class selection, object identity, target
size, rotation, seed, and scene count. Their intended difference was placement
and visibility policy.

## Dataset summary

| Arm | Synthetic scenes | Synthetic boxes | Density median | Density q95 | Median box-area fraction |
|---|---:|---:|---:|---:|---:|
| A0 real train | 0 | 0 | 1 | 5 | 0.012298 |
| A1 naive | 200 | 51,416 | 255 | 295 | 0.001773 |
| A2 VA-DCP | 200 | 51,389 | 255 | 295 | 0.001750 |

A0 contains 8,011 train images and 20,959 real boxes. Therefore the synthetic
boxes dominate the mixed A1/A2 training arms, while their median area is about
one seventh of the real-train median. This is a substantial domain and
optimization shift, not a small augmentation perturbation.

## Validation result

| Arm | mAP50--95 | Recall | Worst AP |
|---|---:|---:|---:|
| A0 | 0.4078 | 0.6021 | approximately 0 |
| A1 | 0.3539 | 0.5362 | approximately 0 |
| A2 | 0.3348 | 0.5301 | 0.0004 |

Paired deltas:

- A1 minus A0 mAP50--95: **-0.0538**;
- A2 minus A0 mAP50--95: **-0.0730**;
- A2 minus A1 mAP50--95: **-0.0192**.

Observed class-level warning:

- `biji_muda` had AP approximately zero in every arm;
- `biji_bertutul_tutul` fell from 0.3832 in A1 to 0.0881 in A2.

## Decision

The original augmentation gate failed. No extra seeds are authorized for that
protocol.

This result establishes only that adding the current dense simulators to a
sparse source-domain training set reduced performance on sparse real
validation. It does **not** establish that A1 or A2 is an invalid simulator for
a real dense 300 g scene, because no matching real dense validation benchmark
exists.

A1 and A2 are retained as simulators. Their next authorized use is:

1. internal paired-quality auditing;
2. analysis of visibility, overlap, class prior, scale, and label-risk proxies;
3. eventual comparison on an identity-independent real dense benchmark.

Test remains locked. Claims about conveyor readiness, physical 300 g mass,
photorealism, and real-dense accuracy remain prohibited.
