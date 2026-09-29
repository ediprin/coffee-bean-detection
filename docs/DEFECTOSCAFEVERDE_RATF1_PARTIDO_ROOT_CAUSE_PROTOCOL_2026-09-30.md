# DefectosCafeVerde RATF1 `partido` Root-Cause Protocol

Date frozen: 2026-09-30  
Status: frozen after the failed seed-42 screen and before this audit  
Training: forbidden  
Evaluation: validation only; test remains locked

## Question

RATF1 reduced `partido` AP50-95 by 2.15 percentage points despite small gains
for `agrio`, `concha`, and `negro`. Is this regression caused by loss of box
accessibility, the active texture-score residual at inference, or a changed
classification representation learned during RATF1 training?

## Endpoints

- matched seed-42 50-epoch `D0DIRECT`;
- the completed seed-42 50-epoch `RATF1` checkpoint in its native active form;
- the same RATF1 checkpoint with every class-residual projection set to zero
  in memory. No weights are saved and no training occurs.

The D0 and RATF result/checkpoint SHAs, dataset-audit SHA, official pretrained
SHA, native config, training schedule, seed, validation split, and test lock
must match their frozen contracts.

## Measurements

For every validation object labelled `partido`, report:

- raw top-500 proposal accessibility and matched recall at IoU 0.5;
- final detections at confidence 0.001;
- localization-conditioned class accuracy and wrong destinations;
- maximum localized true-class and rival scores, their margin, true-class
  rank, and the strongest rival class;
- equality of raw localization between active and zero-residual RATF1.

## Attribution

- lower raw accessibility than D0 indicates localization/shared-feature damage;
- improvement after zeroing the residual implicates inference-time texture
  fusion;
- zero-residual RATF1 remaining below D0 implicates the training path;
- unchanged object-level Top-1 with lower AP indicates score ranking rather
  than localization or hard classification failure.

This is a diagnostic attribution study, not model selection. It authorizes no
new training, test access, or superiority claim.
