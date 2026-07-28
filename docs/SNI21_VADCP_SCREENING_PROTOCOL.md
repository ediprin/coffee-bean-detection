# Frozen protocol: SNI-21 VA-DCP screening

**Version:** 1.0

**Frozen:** 28 July 2026

**Status:** validation-only; training has not started and test remains locked

## Research question

Does visibility-aware dense copy-paste improve a flat 21-class YOLO26n detector
on real validation images, beyond both real-only training and ordinary
copy-paste?

This is a data-augmentation ablation. It is not an architecture comparison and
does not establish performance on real 300 g scenes.

## Fixed detector and label space

- Detector: pretrained `yolo26n.pt`.
- Task: object detection with 21 flat SNI-derived classes.
- Input: 640 px.
- Loss, optimizer, and native Ultralytics augmentation remain identical across
  arms.
- Normal beans remain an explicit detected class (`biji_normal`), not
  background.
- Hierarchical supervision, P2 heads, attention, HBP, and alternative
  detectors are excluded from this experiment.

The class order is taken only from the materialized `data.yaml`. No class may
be silently removed or reordered.

## Dataset authority

The real dataset must be the output of
`coffee_detector.prepare_sni_fullscene` whose audit reports:

- 8,011 train images;
- 416 validation images;
- 451 locked test images;
- all 21 classes in every split;
- zero identity group or exact-duplicate component crossing splits;
- `TRAINING_READY=True`.

The crop library used by A1/A2 must use only `generated_split=train` from
`coffee-sni-instance-crop-v1`. A crop derived from real validation or test is
forbidden.

Validation and test are always real-only. Synthetic images may only be added
to train.

## Arms

| Arm | Training data | Purpose |
|---|---|---|
| A0 | real train only | detector baseline |
| A1 | identical real train + ordinary copy-paste | control for extra synthetic data |
| A2 | identical real train + VA-DCP | isolate visibility-aware placement |

A1 and A2 each contain exactly 2,000 synthetic train scenes. They use:

- the same selected crop identities;
- the same class draws;
- the same target sizes, rotations, and scene counts;
- the same plain-background policy;
- the `source_empirical` composition policy;
- 220–300 placed objects per 1,024 px scene;
- seed 42 during screening.

Only the placement/visibility policy differs. A1 uses ordinary placement; A2
uses the frozen `sni_spread` VA-DCP policy:

- 70% clear, 25% mild, 5% severe, and 0% extreme target visibility;
- spread scenes only;
- support overlap range 0–1.5%;
- visibility-controlled fraction 8%;
- minimum retained visibility 10%.

The term “300 g” is not used as a mass claim. The 220–300 range is a visual
scene-count hypothesis until measured real 300 g photographs are available.

## Screening training

Use:

- configs `A0_yolo26n_screen.yaml`, `A1_yolo26n_screen.yaml`, and
  `A2_yolo26n_screen.yaml`;
- 10 epochs;
- batch 16;
- patience 10;
- seed 42;
- the same accelerator and software environment for all arms;
- resumable checkpoints, but no checkpoint may be selected using test results.

Screening evaluates **validation only**. The runner defaults to
`--evaluation-split val` and rejects test unless `--open-test` is also passed.

## Frozen metrics

Primary metric:

- validation mAP50–95.

Required secondary metrics:

- mAP50, precision, and recall;
- AP50–95 for every class and worst-class AP50–95;
- count MAE and signed count bias;
- exact-count accuracy;
- results stratified by object count per image;
- parameters, FP32 size, batch-1 latency, and FPS.

The count-density strata must be derived once from real-train object-count
quantiles and then applied unchanged to validation. They may not be chosen
after viewing model results.

## Advancement gate

The scientific contrast is **A2 versus A1**, not A2 versus A0.

A2 advances from seed-42 screening only when all conditions hold:

1. validation mAP50–95 is higher than A1;
2. validation recall is not lower than A1 by more than 1.0 percentage point;
3. worst-class AP50–95 is not lower than A1 by more than 1.0 percentage point;
4. count MAE is lower than A1 on the highest real-validation density stratum;
5. dataset and artifact audits report no leakage or incomplete output.

A1 versus A0 is reported to show whether ordinary copy-paste itself helps.
Failure of A2 versus A1 means VA-DCP is stopped; a gain over A0 alone is not
sufficient.

If A2 passes, repeat the unchanged experiment for seeds 123 and 2026.
Confirmation requires a positive mean A2-minus-A1 mAP50–95 delta and passage
of conditions 2–4 on at least two of three seeds. No threshold or policy may be
retuned between seeds.

## Test opening rule

Test remains locked throughout screening and three-seed confirmation. It may
be opened once only after:

1. the validation gate is completed;
2. the final arm and all hyperparameters are frozen;
3. efficiency measurement settings are frozen;
4. the opening decision is recorded in the result log.

The final test report must include A0, A1, and A2, even if A2 fails validation.
Existing test results must never be used to modify this protocol.

## Claim boundary

A successful result supports:

> VA-DCP improved a YOLO26n SNI-21 detector under the audited real split and
> controlled synthetic-train augmentation.

It does not support:

- validated performance on real 300 g samples;
- conveyor robustness;
- generalization to unseen farms, cameras, or illumination;
- physical mass estimation.

Those claims require a separately collected, identity-independent real
benchmark.
