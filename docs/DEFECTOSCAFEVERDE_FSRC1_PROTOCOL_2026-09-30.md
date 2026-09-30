# DefectosCafeVerde FSRC1 Frozen-D0 Protocol

Status: frozen before training. Date: 2026-09-30.

## Research question

Can fixed, spatially aligned frequency information improve detection-score
ranking without changing the already-strong native detector, its boxes, or its
predicted classes?

The question follows directly from the RATF1 failure analysis. RATF1 preserved
object accessibility and improved localization-conditioned top-1 recognition,
but joint training increased duplicate and other-class false positives and
reduced `partido` AP. Zeroing its inference residual did not recover D0 AP,
showing that the dominant damage was learned along the joint training path.

## Candidate

`FSRC1` means Frequency-Guided Selective Reliability Calibration.

1. The completed seed-42 `D0DIRECT` detector is loaded and frozen.
2. A fixed two-level stationary Haar transform extracts six luminance-detail
   maps (horizontal, vertical, and diagonal at each level). It has no trainable
   parameter and does not replace RGB.
3. For each existing D0 candidate, ROIAlign summarizes spectral mean, standard
   deviation, and maximum. Native confidence, normalized geometry, overlap
   with a higher-confidence same-class candidate, and class identity complete
   the descriptor.
4. A 1,281-parameter MLP estimates candidate reliability.
5. Reliability can only suppress confidence within a fixed lower bound:

   `score' = score * (0.25 + 0.75 * reliability)`.

The candidate cannot create a box, move a box, change a class, or increase a
confidence score. Disabling the calibrator returns D0 predictions exactly.

## Training

- Detector: frozen completed D0DIRECT seed 42; no detector optimizer exists.
- Calibrator data: deterministic, augmentation-free predictions on the
  development `train` split only.
- Candidate labels: Ultralytics-compatible class-aware one-to-one matching at
  IoU 0.50.
- Loss: balanced reliability BCE, within-class pairwise ranking, and positive
  preservation.
- Calibrator schedule: 20 fixed epochs, AdamW, learning rate 0.001,
  weight decay 0.0001, seed 42.
- No validation-based early stopping or hyperparameter selection.

The expensive train-candidate cache and completed calibrator checkpoint are
contract-checked and reusable after interruption.

## Evaluation and gates

Evaluation uses the untouched validation split and the calibrated evaluator
already used for D0 and RATF. The D0 endpoint must reproduce historical Macro,
Bottom-3, and Worst-class mAP50-95 within 0.2 point.

All structural gates must pass:

- D0 state tensors remain unchanged;
- candidate boxes and classes are bitwise identical to D0;
- candidate scores never increase;
- all 12 validation classes are present;
- test is absent and unopened.

The seed-42 decision is reported without hiding trade-offs:

- strong Macro route: at least +0.2 point Macro with Bottom-3 and Worst drops
  no larger than 0.1 point;
- lower-tail route: Macro drop no larger than 0.1 point, Bottom-3 gain at least
  +0.5 point, and Worst not lower;
- otherwise a strict Pareto improvement is retained as exploratory;
- mixed gains and losses are reported as a Pareto trade-off, not silently
  rejected or promoted.

No additional seed or test evaluation is authorized by this protocol.

## Claim boundary

One seed is a screening result. FSRC1 may support a frequency-guided ranking
claim only after a predefined promotion route passes and a separate paired
confirmation protocol is frozen. The experiment does not claim improved
localization because coordinates are deliberately unchanged.
