# Faruq-v3 AF2 Reused-Test Post-Hoc Result

Date: 2026-08-21

Status: **completed -- POSTHOC_DIRECTION_POSITIVE**

Scientific status: **REUSED_TEST_POSTHOC_NOT_LOCKED_CONFIRMATION**

## Scope

The original AF2 checkpoints at seeds 42, 123, and 2026 were evaluated on the
same 129-parent Faruq-v3 test package previously consumed by the ACMC study.
The three historical D0FT reports were reused after their test-manifest SHA was
verified against the archive. No training, threshold selection, checkpoint
selection, or tuning was performed.

Because this test had already been opened for another model study, the result
is descriptive post-hoc corroboration. It is not a new untouched locked-test
confirmation.

## Aggregate result

| Metric | D0FT mean | AF2 mean | Mean delta | Minimum seed delta | Improved seeds |
|---|---:|---:|---:|---:|---:|
| Macro mAP50-95 | 86.31% | **88.33%** | **+2.02 points** | +1.00 | 3/3 |
| Bottom-3 mAP50-95 | 72.85% | **76.49%** | **+3.65 points** | +1.44 | 3/3 |
| Worst-class mAP50-95 | 69.07% | **71.46%** | **+2.38 points** | +1.05 | 3/3 |

AF2 improved all three reported metrics in every seed. The standard deviation
of Macro was lower for AF2 (0.63%) than D0FT (0.95%), while the Worst-class
standard deviation was higher (3.59% versus 2.43%).

## Per-seed Macro result

| Seed | D0FT | AF2 | Delta |
|---:|---:|---:|---:|
| 42 | 85.81% | 87.67% | +1.86 points |
| 123 | 87.41% | 88.41% | +1.00 point |
| 2026 | 85.72% | 88.93% | +3.21 points |

## Paired-parent bootstrap

The frozen 1,000-iteration bootstrap used the 129 paired parent identities:

- custom Macro point delta: **+1.93 points**;
- 95% interval: **+0.13 to +3.86 points**;
- probability of a positive paired delta: **98.1%**.

The custom bootstrap Macro differs slightly from the evaluator aggregate
(+1.93 versus +2.02 points) because it is recomputed from saved predictions
under the paired-parent resampling procedure. Both estimates have the same
positive direction.

## Per-class boundary

Mean AP improved for 16 of 21 classes, with the largest positive changes for
`biji_muda` (+7.49 points), `biji_hitam_pecah` (+6.26), and
`tanah_batu_ranting_besar` (+6.13). Five classes had negative mean deltas:
`biji_berlubang_satu` (-3.16), `biji_berlubang_lebih_satu` (-2.64),
`biji_coklat` (-1.33), `biji_hitam_sebagian` (-0.39), and `kopi_gelondong`
(-0.22). The result therefore supports aggregate and lower-tail improvement,
not universal per-class improvement.

## Decision and claim boundary

The frozen descriptive status is:

```text
POSTHOC_DIRECTION_POSITIVE
```

This strengthens the evidence that AF2's validation advantage persists on the
available Faruq test package. It does not convert the package into an untouched
AF2 test, create a new confirmatory significance gate, or authorize further
tuning. The thesis must retain the explicit limitation that no new untouched
in-domain test confirmation exists.

## Artifacts

- Protocol:
  `docs/FARUQ_V3_AF2_REUSED_TEST_POSTHOC_PROTOCOL_2026-08-21.md`
- Drive summary:
  `experiments/faruq-v3-af2-reused-test-posthoc-v1/af2_reused_test_posthoc_summary.json`
- Summary SHA256:
  `57d2e25682df1070a812f5ff0cb6ba2aa02ebc77717cab142f9e99388183c805`
- Test-manifest SHA256:
  `e0655a584d947d98535b94b55d8a109a267ceecfc22a7094204060cefd08b9a6`
- Training executed: `false`
- Test accessed: `true`
- Further tuning authorized: `false`
