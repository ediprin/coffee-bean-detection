# Faruq-v3 DC2 Predicted Raw-Crop Screening Protocol

Date frozen: 2026-08-07
Revision: v2, frozen before any DC2b screening result was consumed
Stage: broad candidate search / predicted-box local stream
Candidate family: `DC2 predicted raw RGB local classifier`
Branch: `agent/dc2-predicted-raw-crop-screening`

## Research question

After the GT-box raw-crop screen establishes that raw-object re-encoding is worth
keeping, does the local-stream signal survive when the crop comes from the
detector's **predicted box**, and does a dedicated raw-RGB local classifier
improve fine-grained classification relative to the native detector decision on
the same matched objects?

## Relationship to DC2

Zheng et al. (IEEE TIP, 2025) use detector output to crop the original image and
re-encode the object with a dedicated local classifier. This arm transfers that
specific mechanism to the coffee detector. It is still **not the complete DC2
network** because global/local multi-stream feature aggregation (MSFA) is
deliberately deferred to a later arm.

The scientific distinction from earlier failed ROI refinement remains fixed:
this local classifier receives the original RGB pixels inside a predicted box,
not ROIAlign features that were already compressed by the detector backbone.

## Frozen prerequisites

1. Faruq-v3 grouped development dataset only; `test/` must be absent.
2. Seed is fixed to 42 for screening.
3. DC2a raw-crop report must use protocol
   `faruq-v3-dc2-raw-crop-resolution-search-v1`, have decision
   `RETAIN_DC2_LOCAL_STREAM`, next action
   `AUTHORIZE_PREDICTED_RAW_CROP_LOCAL_STREAM_INTEGRATION`, and report that test
   images were not accessed.
4. Detector checkpoint is frozen for the whole extraction and is recorded by
   SHA-256. The intended first run is the selected ACMC1 seed-42 checkpoint.
5. DC2a is a **mechanistic prerequisite only**. Its coffee-validation-selected
   `best_resolution` is recorded for audit but is not reused in DC2b.

## Resolution freeze and anti-leakage rule

DC2b uses exactly **128 x 128** input crops.

This value is frozen independently of the coffee validation result. The reason
is methodological: DC2a compared 32/64/128/224 on the development validation
split. Reusing the winning coffee resolution and then evaluating DC2b on that
same validation split would carry a validation-adaptive choice into the next
candidate screen.

Therefore:

- `PAPER_FROZEN_RESOLUTION = 128` is fixed in code before any DC2b result is
  consumed;
- the value follows the fixed crop size used in the later DC2 experiments after
  the paper's own crop-size ablation;
- DC2a's observed `best_resolution` is written into the output only as
  `dc2a_best_resolution_observed_but_not_reused`;
- no DC2b resolution search is permitted.

This does not make validation an independent test set; DC2b remains a
**development-validation screening experiment**.

## Predicted-box construction

- detector inference image size: 640;
- confidence threshold parameter: 0.001;
- predictor IoU parameter: 0.70; native YOLO26 end-to-end postprocessing is left
  unchanged rather than introducing a new class-specific postprocessing rule;
- maximum detections per image: 300;
- GT/prediction matching: class-agnostic one-to-one greedy matching by IoU;
- match threshold: IoU >= 0.50;
- predicted class is **not** used to decide whether a target is matched;
- each matched record stores GT class, GT box, predicted box, predicted class,
  confidence, and matched IoU;
- train and validation caches are separate and checkpoint-hash keyed;
- train and validation caches must resolve to the same detector checkpoint hash.

Class-agnostic matching prevents detector classification errors from removing
exactly the hard samples that the local classifier is supposed to study.

## Paired local-stream control

The same matched object identities are used for two local classifiers:

- `gt`: exact GT raw-RGB crop;
- `predicted`: detector-predicted raw-RGB crop.

Both use the same MobileNetV3-Small ImageNet initialization, fixed 128 x 128
crop resolution, exact context factor 1.0, augmentation, optimizer, schedule,
and seed. The GT-matched arm is only an information-retention control; the
actual candidate is the predicted-crop arm.

A cache/experiment signature contains the protocol version, detector hash,
fixed resolution, detector inference parameters, and matched-record counts.
Existing local checkpoints/summaries are reused only when that signature
matches; otherwise they are treated as stale and retrained.

## Fixed training setup

- local backbone: torchvision MobileNetV3-Small, ImageNet weights;
- classes: 21;
- crop resolution: 128 x 128;
- epochs: 20;
- batch size: 64;
- AdamW learning rate: 3e-4;
- weight decay: 1e-4;
- cosine learning-rate schedule;
- augmentation: horizontal and vertical flip only;
- checkpoint selection: best development-validation Macro-F1 within each
  predeclared paired arm;
- metrics: Accuracy, Macro-F1, Bottom-3 F1, Worst-F1.

These are matched-object classification metrics, not detector mAP.

## Frozen seed-42 gate

Let `N` be native detector classification on matched development-validation
objects, `P` the predicted-crop local classifier, and `G` the GT-crop classifier
on the same matched object identities.

All criteria must pass:

1. train matched coverage >= 90%;
2. validation matched coverage >= 90%;
3. Macro-F1(P) - Macro-F1(N) >= +1.00 percentage point;
4. Bottom-3-F1(P) is not below Bottom-3-F1(N);
5. Worst-F1(P) may not be more than 5.00 points below Worst-F1(N);
6. Macro-F1(P) / Macro-F1(G) >= 0.90;
7. Bottom-3-F1(P) / Bottom-3-F1(G) >= 0.80.

If all pass: `PASS` and authorize a separate DC2 global/local MSFA screen.
Otherwise: `FAIL`; do not infer that all raw-crop approaches are impossible,
only that this frozen predicted-box local-stream arm did not justify escalation.

## Leakage and scope boundaries

- locked test remains untouched;
- no validation confusion pair is used as train-time supervision;
- no class-specific crop rule is introduced;
- no crop-resolution retuning is allowed here;
- DC2a `best_resolution` is explicitly not reused;
- no box-branch retraining occurs in this arm;
- do not call this an end-to-end or full DC2 reproduction;
- MSFA/global-local fusion requires the next separately frozen experiment;
- a PASS is a development-screening authorization, not final confirmation.
