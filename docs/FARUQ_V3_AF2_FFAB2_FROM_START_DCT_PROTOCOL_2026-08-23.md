# Faruq-v3 AF2 + FFAB2 From-Start and DCT Efficiency Protocol

Status: **FROZEN BEFORE NEW TRAINING**  
Date: 2026-08-23  
Test status: **LOCKED**. This protocol uses grouped development train/val only.

## Why this experiment exists

The completed staged FFAB2 experiment established that FFAB2 beats its matched
30-epoch continuation control, but it did **not** establish a causal upgrade over
the pre-continuation original AF2 checkpoint. The new experiment removes that
comparator ambiguity.

No claim is made that the staged experiment was invalid for its original
question. This protocol asks a different and stricter question:

> When AF2 and AF2+FFAB2 start from the same seed-matched D0 checkpoint and use
> the same 50-epoch schedule, does FFAB2 improve the final detector?

Only if that question passes do we test whether a sparse selected-DCT descriptor
can replace FFAB2's full rFFT2 descriptor more efficiently.

## Stage 1 — fair from-start causal comparison

For each seed `42`, `123`, and `2026`:

| Arm | Start | AF2 input frontend | Classification adapter | Epochs |
|---|---|---|---|---:|
| `AF2FS` | seed-matched D0 | original AF2 | none | 50 |
| `AF2FFAB2FS` | **same D0** | original AF2 | bounded FFAB2, rFFT2 high/total ratio | 50 |

Frozen equality requirements:

- same model YAML (`yolo26n-p3`);
- same D0 checkpoint within each seed;
- same AF2 operator parameters;
- same seed, optimizer, image size, batch, workers, patience, augmentation
  schedule and 50 requested epochs;
- FFAB2 starts at exact identity (`alpha=0`);
- FFAB2 residual amplitude remains bounded to ±10%;
- regression/box path always receives the original P3/P4/P5 tensors;
- only the classification path receives the adapted tensors;
- no ROIAlign, second crop, decoded-box feedback, or test access.

The static audit must PASS before either arm trains for that seed.

### Frozen Stage-1 decision

Three-seed PASS requires **all**:

1. mean Macro mAP50-95 gain `>= +0.50 pp`;
2. Macro improves in at least `2/3` seeds;
3. mean Bottom-3 gain `>= +0.50 pp`;
4. Bottom-3 improves in at least `2/3` seeds;
5. mean Worst-class delta is `>= 0`;
6. Worst-class improves in at least `2/3` seeds.

If any criterion fails: `REJECT`, stop the claim that FFAB2 is a direct AF2
upgrade, and do **not** train the DCT stage under this protocol.

If all pass: authorize Stage 2.

## Stage 2 — selected-DCT efficiency replacement

Stage 2 is conditional on Stage-1 PASS.

For each seed, `AF2FFADCTFS` starts from the same seed-matched D0 checkpoint and
uses the exact Stage-1 FFAB2 architecture/schedule except for one field:

```text
AF2FFAB2FS:  descriptor_type = rfft_ratio
AF2FFADCTFS: descriptor_type = dct_selected
```

The DCT arm is **not** a full FcaNet implementation. It is a controlled FFAB2
descriptor replacement. It uses eight fixed, non-learned high-frequency 2-D
DCT-II projections:

```text
(0.00,0.50), (0.50,0.00), (0.50,0.50), (0.00,0.75),
(0.75,0.00), (0.50,0.75), (0.75,0.50), (0.75,0.75)
```

Each channel descriptor is the mean absolute selected DCT response normalized
by channel RMS and bounded by `r/(1+r)`. The basis is fixed and cached; it adds
no trainable parameters. The same FFAB2 scale/bias/alpha and ±10% residual cap
are retained.

This is an efficiency **hypothesis**, not a claim that DCT is faster. Actual
same-device timing decides.

### Frozen Stage-2 decision

Accuracy non-inferiority against `AF2FFAB2FS` requires:

- Macro mean drop no worse than `-0.20 pp`;
- Bottom-3 mean drop no worse than `-0.50 pp`;
- Worst mean drop no worse than `-1.00 pp`.

Efficiency additionally requires:

- selected-DCT adapter median latency at least `20%` lower than rFFT FFAB2 in a
  paired same-device P3/P4/P5 adapter benchmark;
- full 640x640 model median latency is not slower than rFFT FFAB2 on the same
  device;
- parameter counts are exactly equal.

Only if every accuracy and efficiency criterion passes may DCT be called the
retained efficient FFAB2 replacement. Otherwise retain the rFFT FFAB2 design.

## Execution pattern

The Colab notebooks follow the already successful repository pattern:

1. mount Drive;
2. clone the frozen branch;
3. install `ultralytics==8.4.96` plus editable repo;
4. purge/reload `coffee_detector` modules;
5. resolve seed-matched D0 and grouped development bundle;
6. extract development data only; assert no `test/` directory;
7. run static audit in a separate cell;
8. launch direct experiment modules with `subprocess.Popen` and write logs to
   Drive;
9. reuse completed result JSONs and resume only from canonical `last.pt`;
10. run the decision module only when all required seed results exist.

No extra worker layer is introduced.

## Files

- `src/coffee_detector/af2_ffa/dct.py`
- `src/coffee_detector/af2_ffa/from_start_audit.py`
- `src/coffee_detector/experiments/run_faruq_v3_af2_ffa_from_start_arm.py`
- `src/coffee_detector/experiments/run_faruq_v3_af2_ffa_from_start_decision.py`
- `src/coffee_detector/experiments/run_faruq_v3_af2_ffa_dct_decision.py`
- `configs/af2_ffa/AF2FFAB2FS_yolo26n_from_start.yaml`
- `configs/af2_ffa/AF2FFADCTFS_yolo26n_from_start.yaml`
- `tests/test_af2_ffa_from_start_dct.py`

## Evidence boundary

Until Stage 1 is executed, there is **no result** showing that from-start FFAB2
beats original AF2. Until Stage 2 is executed, there is **no result** showing
that the selected-DCT implementation is faster or accuracy-preserving. Do not
promote implementation hypotheses to findings.
