# Faruq-v3 fresh DLRBC seed-42 result — 2026-08-26

## Status

**COMPLETED — STOP_AFTER_SEED42.** The matched linear and quadratic arms
completed 50/50 epochs on grouped Faruq-v3 development validation. Both runs
used fresh optimizers, the same SHA-locked official YOLO26n source, the same
initialized detector state, and no coffee-domain parent. Test was not opened.

Seeds 123 and 2026 and an AF2 factorial are not authorized by this result.

## Paired primary comparison

The frozen causal comparison is `DLRBC_FRESH - LRLIN_FRESH`.

| Arm | Macro mAP50-95 | Bottom-3 | Worst class | Conditional Top-1 | Correct-decision recall |
|---|---:|---:|---:|---:|---:|
| `LRLIN_FRESH` | 81.12% | 62.36% | 60.87% | 46.58% | 33.65% |
| `DLRBC_FRESH` | **82.06%** | **64.22%** | 50.43% | **52.08%** | **40.49%** |
| Delta | **+0.93 pp** | **+1.86 pp** | **-10.45 pp** | **+5.50 pp** | **+6.84 pp** |

Raw top-500 proposal accessibility was identical at 99.81%, so the observed
differences are not explained by raw proposal availability.

Both arms record the same initialized detector-state SHA:

`2591ff254b57e63c9da3b278873a6eb6ab0b998c02976a6eeda18aed06132cd7`

This makes the linear arm the valid matched causal control for the quadratic
interaction.

## Frozen gate

The candidate improved two of the three headline metrics, satisfying the first
criterion. It failed the second criterion because Worst-class mAP50-95 dropped
10.45 points, far beyond the frozen 0.5-point tolerance.

| Criterion | Result |
|---|---|
| Improve at least two headline metrics vs matched linear | PASS |
| No headline drop greater than 0.5 point | **FAIL** |
| Fresh optimizer and no coffee parent | PASS |
| Test remains closed | PASS |

Therefore the prospective decision is:

`STOP_AFTER_SEED42`

The result does not support a stable fine-grained benefit from the quadratic
residual. It raises the average and Bottom-3 metrics but causes a severe
single-class collapse.

## Descriptive cross-protocol context

The previously completed direct-from-pretrained study reported:

| Arm | Macro mAP50-95 | Bottom-3 | Worst class |
|---|---:|---:|---:|
| `D0DIRECT` | 79.90% | 64.57% | 61.77% |
| `AF2DIRECT` | 80.79% | **69.58%** | **66.95%** |
| `LRLIN_FRESH` | 81.12% | 62.36% | 60.87% |
| `DLRBC_FRESH` | **82.06%** | 64.22% | 50.43% |

This table is descriptive only. `D0DIRECT` and `AF2DIRECT` used initialized
detector-state SHA
`f21cfa9fd1e23624494ad57a48bb3fdd878f46a742f57f0d490bee3ac0d08e1a`,
which differs from the fresh DLRBC study. They cannot replace `B0_FRESH` or the
matched linear arm in a causal claim.

## Artifact provenance

- LRLIN checkpoint SHA-256:
  `c8979c6514b0854a90bc49994092fc9c48e34cc65aacc38a6094cb1856c8a91b`
- DLRBC checkpoint SHA-256:
  `0025b68afc9397395136a049d1866b4c9cc2bb1387a624b0dfa0e75ac19fe8f2`
- Raw Drive output root:
  `experiments/faruq-v3-dlrbc-fresh-v1`
- Frozen protocol:
  `docs/FARUQ_V3_DLRBC_FRESH_PROTOCOL_2026-08-26.md`
- Machine-readable evidence:
  `docs/evidence/FARUQ_V3_DLRBC_FRESH_SEED42_2026-08-26.json`

No training or test evaluation was executed while recording this result.
