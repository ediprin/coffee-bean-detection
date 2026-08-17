# Faruq-v3 AF2 Controlled Illumination Robustness Protocol

Status: **frozen before evaluation**  
Date: 2026-08-17

## Question

Does AF2 lose less detection performance than optimization-matched D0FT when
the same validation images receive isolated, geometry-preserving illumination
changes at inference time?

This is an inference-only robustness experiment. It does not train, tune, or
select a new model.

## Data and checkpoints

- Dataset: `faruq-development-v3-grouped`, validation only.
- Validation contains 294 images, 526 objects, and all 21 classes.
- Train images are not evaluated and test remains unavailable.
- Models: paired D0FT and AF2 checkpoints for seeds 42, 123, and 2026.
- Seed 42 is the prospective screen. Seeds 123/2026 may be evaluated only if
  the seed-42 gate passes.

## Controlled conditions

Every source image first follows the same native YOLO decode, letterbox, and
normalization path. The controlled photometric operation is then applied
directly to the BCHW input tensor immediately before inference. `clean` is an
exact tensor clone. No image is re-encoded, and labels are never modified.

| Code | Isolated change |
|---|---|
| `clean` | identity photometric control |
| `dark_ev05` | exposure -0.5 EV |
| `dark_ev10` | exposure -1.0 EV |
| `bright_ev05` | exposure +0.5 EV |
| `bright_ev10` | exposure +1.0 EV |
| `contrast075` | global contrast 0.75 |
| `contrast125` | global contrast 1.25 |
| `warm` | RGB gains 1.12/1.00/0.88 |
| `cool` | RGB gains 0.88/1.00/1.12 |
| `shadow55` | deterministic spatial illumination gradient 0.55-1.00 |

There is no additional resize, crop, affine transform, blur, noise,
compression, cutout, or box modification. Shadow orientation is
deterministically derived from the image name and is identical for both
models and every seed.

## Metrics and estimand

Primary metrics are Macro, Bottom-3, and Worst-class mAP50-95. For model
`M`, condition `c`, metric `q`:

```text
degradation(M,c,q) = q(M,c) - q(M,clean)

robustness_advantage(c,q) =
    degradation(AF2,c,q) - degradation(D0FT,c,q)
```

A positive robustness advantage means AF2 loses less performance than D0FT.
This clean-normalized estimand separates illumination stability from AF2's
ordinary clean-image accuracy advantage.

## Seed-42 gate

AF2 passes the synthetic illumination screen only if all are true:

1. clean Macro is not lower than D0FT;
2. mean Macro robustness advantage is positive;
3. Macro robustness advantage is positive in at least 6/9 stress conditions;
4. the seed-level mean Macro advantage is positive;
5. mean Bottom-3 robustness advantage is non-negative;
6. mean Worst robustness advantage is no lower than -1 point.

If the screen fails, stop without extra seeds or test access.

## Three-seed confirmation gate

When authorized, the same frozen conditions are evaluated for all three
paired seeds. Confirmation requires:

1. AF2 mean clean Macro is not lower than D0FT;
2. mean Macro robustness advantage is positive;
3. Macro advantage is positive in at least 18/27 condition-seed pairs;
4. mean Macro advantage is positive in at least 2/3 seeds;
5. mean Bottom-3 advantage is non-negative;
6. mean Worst advantage is no lower than -1 point.

## Claim boundary

A PASS supports relative robustness to these deterministic synthetic
photometric perturbations. It does not prove robustness to measured lux,
camera exposure, farm, variety, or real shadows. That stronger claim requires
new independent photographs of the same physical objects under controlled
illumination.
