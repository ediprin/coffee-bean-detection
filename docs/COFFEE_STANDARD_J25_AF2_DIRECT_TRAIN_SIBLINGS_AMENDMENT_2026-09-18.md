# Coffee Standard J25 AF2 Direct Train-Siblings Amendment

Status: **frozen before amended training**

Date: 2026-09-18

## Reason for amendment

The completed medoid-only screen revealed that the dataset builder retained a
single representative not only for validation and locked test, but also for
training. This reduced the training set from the available leakage-safe
derivatives to 315 images for 25 classes. The correction is a data-usage
amendment, not a model redesign.

The original medoid-only result remains reported separately. Its validation
metrics are not used to select a new model, hyperparameter, split seed, or
validation composition.

## Frozen corrected dataset

- Source assignment is unchanged: 315 train, 68 validation, and 68 locked-test
  identities at seed 42.
- All three derivatives are retained only when their source identity belongs
  to train.
- Validation retains exactly one medoid per identity.
- Locked test remains manifest-only and is not extracted.
- Corrected train: 695 images and 8,835 boxes.
- Validation: 68 images and 1,164 boxes.
- Locked test: 68 identities and 1,025 boxes.
- All 25 classes remain present in every identity partition.

## Matched rerun

`D0DIRECT` and `AF2DIRECT` restart from the same official `yolo26n.pt` state.
Neither arm reuses the medoid-only checkpoints. Both retain the original
50-epoch schedule, seed 42, image size 640, batch 16, optimizer `auto`, and all
other training settings. AF2 remains a parameter-free frontend.

The original overall and lower-tail decision routes remain unchanged. A fail
stops without extra seeds or locked-test access; a pass authorizes only a
paired three-seed validation confirmation.

Output root: `experiments/coffee-standard-j25-af2-direct-v2`.
