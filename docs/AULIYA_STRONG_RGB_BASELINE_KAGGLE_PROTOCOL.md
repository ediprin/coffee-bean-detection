# Auliya strong RGB baseline — official fold A, Kaggle V1

Status: implementation only, before training. This is the RGB arm of
the registered official-fold frequency screen, not an architecture modification.
Do not compare the validation number directly to the previous B->C pilot.

## Data
Reuse the mounted extracted Kaggle development directory of the completed
V/F/R audits. Verify the exact content fingerprint
12d60838ae810a2841607fd5a09afe6b77c8a48977ea2b9c6bc3a52394d55166
against the audited file inventory and official Zenodo metadata hash.
The original pilot ZIP is not required or claimed byte-verified.
Use the original author fold-A lists in `splits_metadata_results_scripts.zip`
(Zenodo 24346584, script `scripts/training/run_baseline_v4.py`).
Preserve original filenames and list hashes.

| Role | Group B images | Group C images | Total images | Instances |
|---|---:|---:|---:|---:|
| Train | 326 | 350 | 676 | 2130 |
| Validation | 49 | 70 | 119 | 376 |
| Held-out A | 0 | 0 | 1351 | not evaluated |

Counts computed from the official manifest and lists; staged image and label
bytes must match the fingerprint-verified input. The official validation split
is image-level within B/C, and may contain repeated specimens across training
and validation. C was already used in exploratory audits. Do not treat this
validation as independent confirmation or infer a cross-acquisition gain.
Group A is never materialized/read for training or evaluation.

## Training
Native YOLO26s-seg RGB initialized from official v8.4.0 asset (verified hash).
Ultralytics 8.4.210; 21 classes; seed 0; 640 square input setting; batch 16;
80 maximum epochs; patience 25; AdamW, lr0 .001, weight decay .0005,
cosine schedule, four workers. Augmentation exactly follows the author
script and existing official fold protocol. Other defaults are tied to the
pinned Ultralytics version. No reuse of Faruq or pilot checkpoints.
Native best.pt selected by the trainer's validation fitness; retain
optimizer-bearing last.pt and best.pt for resume. Same code/runtime/GPU
contract required to resume. If OOM, stop rather than silently change batch.
Kaggle runtime may differ from authors' runtime; this is an adaptation of the
author protocol, not an exact reproduction of test results.

## Evaluation and next decision
Report validation-only box and mask mAP50–95, clean18 mean, bottom-three AP,
worst AP, all21 mean, classwise AP, and all21 AP50. IDs 18–20 remain in training
and supplementary aggregate; they are excluded from the clean18 mean because
of author-documented specimen reuse. Save native training logs and best/last.
No test-A run, no AF2/HAAR arm, no new loss in this stage.

The next step is a classification-supervision audit on this baseline,
not immediate module/loss promotion. Since earlier R/V/F experiments
used a different B/C split and smaller model, their evidence motivates
hypotheses but must not be numerically compared to this validation result.
Use a separate controlled paired experiment to establish a gain; one
seed and image-level validation are not confirmation.

## Kaggle workflow
Import the new notebook `Auliya_Official_Fold_A_RGB_Kaggle.ipynb`.
Attach the existing extracted `auliya-robusta-sni` dataset only.
Enable Internet and GPU (one T4); the metadata and official small model
asset are fetched and SHA-verified. The old RGB last.pt is not used.
Run All runs preflight and then RGB training. Set AUTHORIZE_TRAINING=False
if only preflight is desired. Save Version to preserve the checkpoint pair.
Attach one previous output directory for resume, not the report-only ZIP.
A report-only archive is produced for analysis, while model checkpoints
remain in the Kaggle working project. No execution on actual Kaggle data
has been claimed by the implementation.
