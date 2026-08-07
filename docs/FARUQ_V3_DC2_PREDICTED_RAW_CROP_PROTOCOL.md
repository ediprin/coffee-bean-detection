# Faruq-v3 DC2 Predicted Raw-Crop Screening Protocol

Date frozen: 2026-08-07
Stage: broad candidate search / predicted-box local stream
Candidate family: `DC2 predicted raw RGB local classifier`
Branch: `agent/dc2-predicted-raw-crop-screening`

## Research question

After the GT-box raw-crop resolution screen, does the local-stream signal survive
when the crop comes from the detector's **predicted box**, and does a dedicated
raw-RGB local classifier improve fine-grained classification relative to the
native detector decision on the same matched objects?

## Relationship to DC2

Zheng et al. (IEEE TIP, 2025) use detector output to crop the original image and
re-encode the object with a dedicated local classifier. This arm transfers that
specific mechanism to the coffee detector. It is still **not the complete DC2
network** because global/local MSFA is deliberately deferred to a later arm.

The scientific distinction from earlier failed ROI refinement remains fixed:
this local classifier receives the original RGB pixels inside a predicted box,
not ROIAlign features that were already compressed by the detector backbone.

## Frozen prerequisites

1. Faruq-v3 grouped development dataset only; `test/` must be absent.
2. Seed is fixed to 42 for screening.
3. DC2a raw-crop summary must have decision `RETAIN_DC2_LOCAL_STREAM`.
4. Crop resolution is inherited from DC2a `best_resolution`; it is **not**
   re-selected in this arm.
5. Detector checkpoint is frozen for the whole extraction and is recorded by
   SHA-256. The intended first run is the selected ACMC1 seed-42 checkpoint.

## Predicted-box construction

- detector inference image size: 640;
- confidence threshold: 0.001;
- prediction NMS IoU: 0.70;
- maximum detections per image: 300;
- GT/prediction matching: class-agnostic one-to-one greedy matching by IoU;
- match threshold: IoU >= 0.50;
- predicted class is **not** used to decide whether a target is matched;
- each matched record stores GT class, GT box, predicted box, predicted class,
  confidence, and matched IoU;
- train and validation caches are separate and checkpoint-hash keyed.

Class-agnostic matching prevents detector classification errors from removing
exactly the hard samples that the local classifier is supposed to study.

## Paired local-stream control

The same matched object identities are used for two local classifiers:

- `gt`: exact GT raw-RGB crop;
- `predicted`: detector-predicted raw-RGB crop.

Both use the same MobileNetV3-Small ImageNet initialization, inherited crop
resolution, exact context factor 1.0, augmentation, optimizer, schedule, and
seed. The GT-matched arm is only an information-retention control; the actual
candidate is the predicted-crop arm.

## Fixed training setup

- local backbone: torchvision MobileNetV3-Small, ImageNet weights;
- classes: 21;
- epochs: 20;
- batch size: 64;
- AdamW learning rate: 3e-4;
- weight decay: 1e-4;
- cosine learning-rate schedule;
- augmentation: horizontal and vertical flip only;
- checkpoint selection: best validation Macro-F1 within each predeclared arm;
- metrics: Accuracy, Macro-F1, Bottom-3 F1, Worst-F1.

These are matched-object classification metrics, not detector mAP.

## Frozen seed-42 gate

Let `N` be native detector classification on matched validation objects,
`P` the predicted-crop local classifier, and `G` the GT-crop classifier on the
same matched object identities.

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
- no box-branch retraining occurs in this arm;
- do not call this an end-to-end or full DC2 reproduction;
- MSFA/global-local fusion requires the next separately frozen experiment.
