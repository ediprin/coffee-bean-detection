# Faruq-v3 AF2_ORIENT + CMC0 Seed-42 Screening Result

Date: 2026-08-23
Status: **STOP — RAW SCREEN FAIL; NO TUNING; NO MULTI-SEED**

## Scope

This document records the completed seed-42 grouped-validation screening of
`AF2_ORIENT + CMC0` against its matched `AF2_ORIENT` parent.

This is an exploratory composition result only. It does **not** establish a
clean causal mechanism because the execution history contains a resume-boundary
anomaly (duplicate epoch 35). The anomaly is preserved explicitly below rather
than hidden.

## Matched parent

Parent result:

`experiments/faruq-v3-af2-isolated-seed42-v1/val_reports/AF2_ORIENT_seed42_result.json`

Parent metrics:

| Model | Macro mAP50-95 | Bottom-3 | Worst |
|---|---:|---:|---:|
| AF2_ORIENT | 88.3260% | 81.3806% | 80.1357% |

## Candidate result

Canonical grouped validation on all 21 classes:

| Model | Precision | Recall | mAP50 | Macro mAP50-95 | Bottom-3 | Worst |
|---|---:|---:|---:|---:|---:|---:|
| AF2_ORIENT + CMC0 | 86.6708% | 80.0335% | 89.4279% | 87.2293% | 79.7854% | 76.4616% |

Candidate minus parent:

| Metric | Delta |
|---|---:|
| Macro mAP50-95 | **-1.0967 pp** |
| Bottom-3 | **-1.5952 pp** |
| Worst | **-3.6741 pp** |

All frozen screening routes fail:

- superiority route: FAIL;
- tail-Pareto route: FAIL;
- raw screening decision: **FAIL**.

The worst class is `biji_berkulit_tanduk` at **76.4616% mAP50-95**.

## Per-class mAP50-95

| Class | mAP50-95 |
|---|---:|
| biji_berkulit_tanduk | 76.4616% |
| biji_berlubang_lebih_satu | 93.4640% |
| biji_berlubang_satu | 81.2945% |
| biji_bertutul_tutul | 84.0974% |
| biji_coklat | 91.6421% |
| biji_hitam | 85.1938% |
| biji_hitam_pecah | 90.0724% |
| biji_hitam_sebagian | 83.1839% |
| biji_muda | 85.3605% |
| biji_normal | 89.4830% |
| biji_pecah | 86.5864% |
| kopi_gelondong | 94.8324% |
| kulit_kopi_ukuran_besar | 89.7510% |
| kulit_kopi_ukuran_kecil | 92.7840% |
| kulit_kopi_ukuran_sedang | 84.3461% |
| kulit_tanduk_ukuran_besar | 81.6002% |
| kulit_tanduk_ukuran_kecil | 84.4051% |
| kulit_tanduk_ukuran_sedang | 83.7217% |
| tanah_batu_ranting_besar | 92.3053% |
| tanah_batu_ranting_kecil | 90.3936% |
| tanah_batu_ranting_sedang | 90.8368% |

No validation class is missing ground truth.

## Latency

Measured batch-1 latency on Tesla T4, 640x640, 10 warmup iterations and 50
measured iterations:

- median: **43.3865 ms**;
- p95: **65.9607 ms**.

## Execution-history anomaly

The postprocessor recorded 51 `results.csv` rows for nominal epochs 1-50, with
`epoch 35` appearing twice:

`..., 33, 34, 35, 35, 36, ..., 50`

Therefore:

- `decision_clean = false`;
- postprocessed decision label: `REVIEW_RESUME_HISTORY`;
- the result must not be represented as a clean confirmation run.

The completed checkpoint is still valid for inspection and canonical validation,
but the duplicated resume boundary blocks a clean causal-confirmation claim.

## Scientific decision

The execution anomaly does not rescue the candidate under the frozen screen:
the observed raw deltas are negative on **all three** target metrics and miss the
pre-registered gates by substantial margins.

Therefore the research decision is:

**STOP `AF2_ORIENT + CMC0`; do not tune this composition on the same validation evidence; do not authorize seeds 123/2026.**

A clean rerun would only be justified for archival/reproducibility hygiene, not
because the current candidate is near the acceptance boundary.

## Artifact identifiers

- candidate checkpoint SHA-256: `3cf6fbc48b9e1c40843e3bb00d2af3493b0296322b2c2f0e823f0680a64a0aa9`
- D0 seed-42 checkpoint SHA-256: `0c458841b84bedce4e0ddada6a5773f6a5ac8a91dad084a4a5f24e89f04e6367`
- AF2_ORIENT parent-result SHA-256: `4fd220498d729548c17b1a3f87f25b1f2b6f4387518edc0947fe437b7685b2cc`
- evaluation split: `val`
- test images accessed: `false`

## Storage note

This run was produced while `save_period: 1` was active and therefore created
large per-epoch checkpoints. Future runs on this branch use `save_period: -1`;
periodic `epoch*.pt` files from this completed failed screen are not required for
the scientific record once the final `best.pt`, result JSON, config, and reports
are preserved.
