# SNI-21 Ontology-Marginal Screening Result

Date: 2026-08-02

Protocol: `faruq-v3-ontology-marginal-screening-v1`

Seed: 42

Evaluation: Faruq-v3 validation only

Test opened: no

## Decision

**FAIL. Stop ontology-marginal v1 without additional seeds or test access.**

| Model | Macro mAP50-95 | Bottom-3 mAP | Worst AP | Proposal accessibility | Conditional top-1 |
|---|---:|---:|---:|---:|---:|
| D0 baseline | 79.97% | 68.72% | 65.09% | 63.69% | 62.99% |
| C0 identity control | 81.78% | 72.79% | 71.50% | 66.73% | 57.83% |
| S0 semantic marginal | 84.06% | 74.46% | 70.96% | 72.24% | 45.53% |

S0 improved Macro mAP by 4.09 points over D0 and 2.28 points over C0. It
also improved proposal accessibility and lower-tail AP. It nevertheless failed
the frozen classification gate: conditional top-1 fell by 17.46 points against
D0 and 12.31 points against C0.

## Raw diagnostic confirmation

The conditional-accuracy collapse is present in the raw counts, not introduced
by table aggregation:

| Model | Targets | Accessible/matched | Correct leaf class | Wrong leaf class | Missed |
|---|---:|---:|---:|---:|---:|
| D0 | 526 | 335 | 211 | 124 | 191 |
| C0 | 526 | 351 | 203 | 148 | 175 |
| S0 | 526 | 380 | 173 | 207 | 146 |

S0 retrieves 45 more targets than D0, but produces 38 fewer correct leaf
decisions and 83 more wrong-class decisions. Frequent S0 confusions include:

- `biji_berkulit_tanduk` -> `kulit_tanduk_ukuran_besar` (23);
- `biji_muda` -> `biji_bertutul_tutul` (17);
- `biji_coklat` -> `biji_bertutul_tutul` (14);
- `kulit_tanduk_ukuran_kecil` -> `kulit_tanduk_ukuran_sedang` (13);
- `biji_pecah` -> `kulit_tanduk_ukuran_besar` (11).

## Interpretation

Ontology marginalization improves coarse semantic mass and detection recall,
but it does not preserve the within-group leaf boundary required by the
official 21-class task. The result is especially important because C0 also
improves mAP: part of the gain comes from additional classification
supervision/weighting rather than semantic grouping alone. S0 adds a further
mAP gain over C0, but exchanges leaf specificity for group-level tolerance.

The mAP and conditional-top-1 results are not mathematically identical
measurements. Ultralytics mAP integrates class-specific precision-recall across
confidence thresholds, while the frozen diagnostic measures the class of
matched detections at its fixed operating configuration. The protocol required
both; therefore the higher mAP cannot override the failed conditional-class
gate.

## Research consequence

Do not tune the auxiliary gain after observing validation, do not run seeds
123/2026, and do not open test. Ontology-marginal v1 remains a documented
negative result: useful for coarse recall, unsuitable as the final flat SNI-21
classifier. Any future structured method must explicitly retain leaf-level
separation rather than only rewarding probability mass within an ontology
group.
