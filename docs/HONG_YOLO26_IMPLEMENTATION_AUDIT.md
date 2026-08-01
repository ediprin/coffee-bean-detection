# Hong-to-YOLO26 implementation audit

Document ID: `HONG-YOLO26-IMPLEMENTATION-AUDIT`

Status: static implementation complete; no GPU training and no test access.

Date: 2026-08-02.

Protocol: `HONG_YOLO26_REPRODUCTION_PROTOCOL.md` v1.2.0.

## Implemented graph

The implementation is a native single-stage YOLO26 detector. It does not use
ROI crops, an external classifier, or a second inference pass.

| Mechanism | Pinned target | Count |
|---|---|---:|
| VQK/KDS/CDS DSConv | `model.1`, `model.3`, `model.17` | 3 |
| SPPF-Attention | `model.9`, before the existing C2PSA | 1 |
| PConv | two spatial blocks on each P3/P4/P5 branch in `cv2`, `cv3`, `one2one_cv2`, and `one2one_cv3` | 24 |

The PConv mapping includes box regression and classification. This corrects
the earlier classification-only draft after visually checking Hong et al.'s
Fig. 1 and Section 4.4. The twelve terminal 1x1 prediction modules are not
replaced and their tensor hashes are checked before and after injection.

## Reconstruction disclosures

- DSConv uses 4-bit VQK simulation and input-channel blocks of 128. Hong does
  not report those two implementation values, so they are frozen reconstruction
  choices rather than claimed verbatim hyperparameters.
- The latent convolution kernel is copied from the pretrained layer. KDS is
  initialized by blockwise least squares; KDS/CDS bias starts at zero and CDS
  scale at one.
- Existing SPPF `cv1`, `cv2`, and max-pooling modules are reused. Channel and
  spatial attention weights are new.
- A dense 3x3 regression convolution cannot be exactly factored into the Hong
  PConv form. Its partial spatial kernel starts as identity, its pointwise
  projection is derived from the source kernel centre, and its BN/activation
  are copied. Classification PConv reuses the source pointwise projection and
  active depthwise diagonals.
- This PyTorch DSConv evaluates the reconstructed floating-point kernel. It
  supports accuracy experiments but does not justify an integer-runtime speed
  claim.

## Verified static gate

The local audit used the pinned YOLO26n YAML, 21 output classes, and the local
official `yolo26n.pt` checkpoint.

| Check | Result |
|---|---|
| Pretrained load before injection | 606/708 checkpoint items transferred by Ultralytics |
| Train output | both `one2many` and `one2one` present |
| Native YOLO detection loss | finite forward and backward |
| Batch-1 and batch-2 inference | PASS |
| Feature levels | P3/P4/P5 shapes preserved |
| State-dict save/reload equivalence | PASS |
| Full model plus optimizer checkpoint/resume equivalence | PASS |
| DSConv after Ultralytics `model.fuse()` | 3/3 remain DSConv |
| Terminal prediction tensor hashes | unchanged |
| Test images accessed | no |

The full transferred model contains 2,590,648 trainable parameters. The local
state dict occupies 10,659,892 bytes. The CPU latency value produced by the
smoke audit is deliberately not used as an efficiency claim; paired T4 timing
is part of the one-seed validation runner.

Repository verification: `155 passed` under the complete pytest suite on
2026-08-02.

## Runnable entry points

- Static audit: `python -m coffee_detector.hong_transfer.audit`
- One-seed fail-fast runner:
  `python -m coffee_detector.experiments.run_hong_yolo26_transfer`
- Colab notebook: `notebooks/Hong_YOLO26_Full_Transfer_Colab.ipynb`

The notebook defaults to `RUN_SCREEN = False`. The static report must be
reviewed before enabling the 50-epoch seed-42 validation screen. A runtime
reset resumes from the `last.pt` stored directly under the shared project Drive.

## Current decision

Implementation gate: **PASS**.

Research result: **not available yet**. No statement about accuracy, superiority,
or transferability is permitted until `HF_seed42` is trained and all frozen
validation, conditional-classification, lower-tail, operational, and latency
criteria are evaluated.
