# Coffee Standard J25 Fixed-Holdout Feasibility Audit

> **RETRACTED — INVALID SOURCE-IDENTITY PREMISE.** This result must not be
> used. The audit treated the filename prefix before `.rf.<hash>` as an
> authoritative source identity. The thesis and original `data_aug_11.zip`
> establish 451 raw photographs, whereas this rule collapsed the export to 267
> components. Basenames are reused, so the rule over-merged unrelated photos.
> The executable audit now fails fast. See
> `COFFEE_STANDARD_J25_THESIS_PROVENANCE_RESULT_2026-09-18.md`.

Status: **FAIL â€” FIXED 25-CLASS HOLDOUT NOT ADEQUATELY SUPPORTED**  
Date: 2026-09-18

## Question

The text below is retained only as a record of the withdrawn analysis.

Can the 253 eligible, leakage-controlled J25 v2 source identities be divided
into train/validation/test while retaining enough independent source identities
for meaningful per-class held-out evidence?

This is a model-free audit. No training or model test evaluation is performed.

## Frozen interpretation rule

Object count is not treated as independent support. Many boxes from one source
photograph remain one source identity. The hard gate requires at least two
independent identities for every class in both validation and test, while five
is reported as a recommended planning target. Two is intentionally the weakest
non-degenerate gate: a held-out class represented by one photograph cannot
measure between-capture variation.

## Result

The global theoretical ceiling is **one independent identity per held-out
split**. This is a mathematical limit after reserving at least one identity for
train, not a failure of the split optimizer.

| Class | Total objects | Source identities | Maximum common identities in val and test |
|---|---:|---:|---:|
| Biji Berkulit Ari | 193 | 4 | 1 |
| Kerikil | 138 | 4 | 1 |
| Kopi Gelondong | 85 | 4 | 1 |

For any class with four identities, keeping one identity in train leaves only
three for validation and test. Therefore both held-out splits cannot each
receive two independent identities. Several additional classes have only five
or six identities and remain weak even when the hard gate is relaxed.

The current v2 split also fails a ten-object descriptive floor for some held-out
classes. More importantly, a split can satisfy a box-count floor by putting many
boxes from a single photograph into a held-out split; this does not repair the
independence problem.

## Decision

`FAIL_FIXED_HOLDOUT_PRIMARY_BENCHMARK`

J25 v2 remains useful as a paper-backed, leakage-controlled development or
external dataset, but the current 253 identities cannot support a conventional
25-class train/validation/test benchmark with stable per-class Bottom-3/Worst
claims.

Next options are:

1. acquire additional independent source photographs for the rare classes;
2. use grouped cross-validation and report fold coverage/uncertainty rather
   than one fixed per-class holdout;
3. reduce or merge ontology classes only with authoritative SNI/ICO and author
   justification, never to improve model metrics post hoc.

Training remains unauthorized, and the unresolved 451/993/2,000-image
provenance discrepancy remains separate and visible.
