# SNI-21 Structured-Target Support Audit Result

Date: 2026-08-02

Protocol: `sni21-structured-target-support-audit-v1`

Dataset: Faruq-v3 grouped development (`train` and `val` only)

Training executed: no

Inference executed: no

Test images accessed: no

## Decision

**AUDIT COMPLETE — NO TRAINING AUTHORIZATION.** Every ontology value clears the
frozen instance/group support thresholds. The data are therefore not blocked
by sample count for an initial structured-target protocol. This result does
not prove that every target is visually observable from RGB.

| Task | Values | Min train instances/groups | Min val instances/groups | Statistical support | Semantic gate |
|---|---:|---:|---:|---|---|
| entity family | 5 | 139 / 76 | 25 / 18 | yes | eligible |
| primary condition | 13 | 139 / 76 | 25 / 10 | yes | eligible |
| original class | 21 | 136 / 75 | 24 / 10 | yes | eligible baseline |
| positive flag | 13 | 139 / 76 | 25 / 10 | yes | positive-only partial supervision |
| surface extent | 5 | 138 / 79 | 24 / 10 | yes | eligible, resolution-sensitive |
| integrity fraction | 2 | 148 / 90 | 26 / 10 | yes | eligible, shape-sensitive |
| relative completeness | 4 | 277 / 153 | 49 / 26 | yes | domain-expert review required |
| hole count | 3 | 138 / 84 | 24 / 10 | yes | eligible, visibility-sensitive |
| physical size | 3 | 138 / 77 | 24 / 16 | yes | blocked until calibrated scale exists |

The support thresholds were 50 instances and 25 independent groups on train,
and 10 instances and 10 independent groups on validation. No ontology value
failed those thresholds.

## Interpretation

The audit separates two questions that must not be conflated:

1. There are enough labelled examples and independent source groups to define
   the structured targets in development.
2. The labels alone do not make every target recoverable from RGB.

`physical_size_mm` remains unusable without a scale reference even though its
three values have ample support. `relative_completeness` has ample support but
still needs an SNI expert to confirm that the visual labels consistently encode
fractions of an intact covering. Positive flags cannot supply arbitrary
negative labels because omitted attributes are unknown.

The next step is a domain review of the two semantic gates, followed by a
frozen flat-versus-structured model protocol. Architecture training remains
blocked until that protocol is approved.
