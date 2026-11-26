# Auliya v2 baseline reproduction and task bridge

Status: implementation scaffold only; real data preparation, geometry review and GPU runs pending.
This branch does not modify the thesis proposal or any old Faruq experiment.

## Inputs

Use the original archive `images_labels_original_2148.zip`, extracted into a directory containing
`train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels`.
Also supply the original `splits_metadata_results_scripts.zip` (SHA256 checked by the runner).

Author package audit: 2148 manifest records; 2146 annotated images in each fold; 3858 instances.
Fold A: 676/119/1351 train/val/test; B: 1506/265/375; C: 1468/258/420.
Original author lists are preserved. The reported held-out acquisition-group separation is
not a guarantee of individual specimen independence: classes 18–20 reuse material,
and train/validation selection is random within remaining groups. Acquisition group and specimen
effects are confounded. Report 21-class and 18-class AP separately; do not claim fully disjoint data.

The previous Mendeley v1 upload has no labels and is not a replacement for the v2 archive.
The prior Faruq data may share lineage; never initialize these runs using coffee-trained checkpoints.

## Baselines

| Arm | Model | Purpose |
|---|---|---|
| B0 | yolo26s-seg.pt | Reproduce author YOLO26s-Seg |
| B1 | yolo26s.pt | Detection-only comparator at s scale |
| B2 | yolo26n.pt | Detection baseline matching thesis model family |
| B3 | not implemented | Future matched fine-grained head treatment after diagnosis |

Use an isolated runtime with Ultralytics 8.4.210 and, if compatible with the GPU, PyTorch 2.5.1
as specified in the author README. Do not upgrade the older repository's pinned 8.4.96 environment
in-place; existing custom heads need a separate compatibility audit.
All three arms use author training parameters: 640, 80 epochs, batch 16, AdamW, lr0 .001,
weight decay .0005, cosine LR, patience 25, original augmentations, seeds 0/1/2.
Other defaults inherit the pinned library. B1/B2 are new baselines, not published author results.

## Usage

From the repository root in the isolated environment:

```bash
python scripts/auliya_baseline.py prepare --fold C --images-root /path/to/originals --metadata-zip /path/to/splits_metadata_results_scripts.zip --output /path/to/foldC
# Review source and converted annotations visually, especially EXIF orientation.
python scripts/auliya_baseline.py train --fold C --output /path/to/foldC --arm B0 --seed 0 --geometry-reviewed
python scripts/auliya_baseline.py train --fold C --output /path/to/foldC --arm B1 --seed 0 --geometry-reviewed
python scripts/auliya_baseline.py train --fold C --output /path/to/foldC --arm B2 --seed 0 --geometry-reviewed
python scripts/auliya_baseline.py evaluate --fold C --output /path/to/foldC --arm B2 --seed 0 --split val
```

Fold C and seed 0 are examples of smoke-run commands, not an optimized or designated holdout.
The runner creates separate detection and segmentation copies; ensure sufficient disk capacity.
It audits labels/counts/paths and exact cross-split image hashes; it does not independently establish
physical identity or mask geometry, nor validate data against the author visually.
The evaluation on test is separate: use `--split test --frozen-test` only after model and evaluation
protocol are locked. For the author reproduction report, run B0 across A/B/C and 0/1/2 and compare
box and mask AP against the author CSV; for thesis development keep test outcomes out of model
selection. Do not tune on any already evaluated held-out group and then describe it as untouched.
Before the final thesis protocol, decide development/confirmation grouping explicitly; if selecting
methods using cross-fold results, do not report them as independent confirmation.

Report checkpoints, training args, exact runtime, per-class AP, missing-class support, 18-class
AP and all 21-class AP. Additional fine-grained diagnosis (class-agnostic AP, matched classification
accuracy, error decomposition) and integration of AF2/IGEM are deliberately deferred until a valid
baseline and independent development protocol are established.

Current blockers: complete original archive not yet supplied; no GPU runs or runtime compatibility
tests have been performed. No Scopus publication status claim.
