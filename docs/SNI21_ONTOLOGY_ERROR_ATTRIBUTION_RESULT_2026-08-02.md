# SNI-21 Ontology Error Attribution Result

Date: 2026-08-02

Protocol: `faruq-v3-ontology-error-attribution-v1`

Seed: 42

Evaluation: Faruq-v3 validation only

Training executed: no

Test opened: no

## Decision

**STOP the structured ontology-marginal direction for this protocol. Do not
tune its gain, run additional seeds, or open test.**

The attribution does not support the hypothesis that S0 mainly exchanges
leaf-level accuracy for semantically acceptable errors between close siblings.
Its additional wrong-class decisions are broader and are dominated by errors
that cross the configured entity-family boundary.

## Wrong-class attribution

| Model | Wrong class | Same primary condition | Same entity family only | Cross entity family |
|---|---:|---:|---:|---:|
| D0 baseline | 124 | 44 | 52 | 28 |
| C0 identity control | 148 | 45 | 48 | 55 |
| S0 semantic marginal | 207 | 38 | 91 | 78 |

Relative to D0, S0 changes the error categories as follows:

| Error category | S0 - D0 |
|---|---:|
| Same primary condition | -6 |
| Same entity family only | +39 |
| Cross entity family | +50 |

The combined increase inside the configured ontology is `-6 + 39 = +33`,
whereas the increase outside the entity family is `+50`. Both categories
worsen, but cross-family errors contribute the larger increase.

## Largest directional increases

| Expected class | Predicted class | D0 | S0 | Delta |
|---|---|---:|---:|---:|
| `biji_berkulit_tanduk` | `kulit_tanduk_ukuran_besar` | 10 | 23 | +13 |
| `kulit_kopi_ukuran_besar` | `kulit_tanduk_ukuran_besar` | 0 | 11 | +11 |
| `biji_coklat` | `biji_bertutul_tutul` | 4 | 14 | +10 |
| `biji_pecah` | `kulit_tanduk_ukuran_besar` | 1 | 11 | +10 |
| `biji_muda` | `biji_bertutul_tutul` | 11 | 17 | +6 |
| `kulit_tanduk_ukuran_sedang` | `kulit_tanduk_ukuran_besar` | 2 | 8 | +6 |
| `biji_hitam` | `biji_coklat` | 2 | 7 | +5 |
| `biji_hitam_pecah` | `biji_hitam_sebagian` | 3 | 8 | +5 |
| `biji_hitam_sebagian` | `biji_normal` | 0 | 5 | +5 |

Several increases cross meaningful object or material boundaries, including
bean-to-parchment and coffee-skin-to-parchment errors. S0 therefore is not
merely group-correct but leaf-wrong; its auxiliary objective destabilizes
fine-grained leaf discrimination more generally.

## Relation to the screening result

The earlier screening showed that S0 improved Macro mAP50-95 from 79.97% to
84.06% and proposal accessibility from 63.69% to 72.24%, while conditional
top-1 accuracy fell from 62.99% to 45.53%. The raw counts showed 45 additional
accessible targets but 38 fewer correct leaf decisions and 83 additional
wrong-class decisions.

This attribution explains why the mAP gain cannot justify continuing the
method: the classification collapse is not concentrated in an intended,
semantically tolerable subset of the label space. The result remains a useful
negative finding about objective mismatch between coarse semantic tolerance
and the official flat 21-class decision.

## Research consequence

- Keep D0 as the valid detector baseline for subsequent evidence synthesis.
- Do not continue ontology-marginal v1 with gain tuning, a hierarchical head,
  additional seeds, or test evaluation.
- Retain the ontology for dataset audit, error reporting, and operational
  aggregation; this result only rejects its current use as an auxiliary
  training objective.
- Do not claim that structured ontology is universally ineffective. The
  supported claim is limited to this loss, dataset split, model, and frozen
  protocol.

The next action is evidence consolidation and thesis-positioning from the
completed controlled studies, not another unmotivated architecture trial.
