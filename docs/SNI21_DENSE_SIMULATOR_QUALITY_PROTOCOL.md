# SNI-21 dense simulator quality audit

**Version:** 1.0
**Frozen:** 30 July 2026
**Status:** internal simulator audit; no detector training and no test opening

## Purpose

A1 and A2 are retained as simulators of dense scenes containing approximately
220--300 objects. The prior YOLO26n screening evaluated them on sparse real
validation data and therefore answered only a source-domain transfer question.
It did not validate either simulator against a real dense scene.

The recorded screening result and its claim boundary are in
`SNI21_VADCP_PILOT_RESULT_2026-07-30.md`.

This audit asks a narrower question:

> Are A1 and A2 internally valid paired simulators, and does A2 create a
> measurable risk that a fine class label remains attached after its
> discriminative visual cue has been occluded?

## Frozen arms

- **A1:** ordinary dense copy-paste.
- **A2:** visibility-aware dense copy-paste (VA-DCP).
- Scene count, random seed, selected crop identities, class draws, target
  geometry, and transforms must be paired.
- Placement, overlap, z-order, and resulting visibility are allowed to differ.

The audit matches scenes with `generation_seed` and instances with their
`z_order`. It verifies the sequence of `source_asset_id`, category, target
aspect ratio, and achieved aspect ratio. Full-mask area equality is diagnostic
only because placement at a canvas border can clip an otherwise identical
transformed cutout.

## Required outputs

The JSON report contains:

1. paired-scene and paired-instance contract checks;
2. A1/A2 density, visible/full bounding-box area, visibility, and ignored-label
   summaries;
3. total-variation and Jensen--Shannon distances between class priors;
4. paired A2-minus-A1 visibility deltas overall and per class;
5. a semantic-label-risk proxy for visibility-sensitive classes;
6. optional synthetic-box dominance and real-to-synthetic scale ratios.

The default visibility-sensitive classes are `biji_muda` and
`biji_bertutul_tutul`, because these were the observed failure classes in the
pilot. Additional classes must be declared before inspecting their result.

The frozen visibility thresholds are:

- moderate-risk proxy: visible instance area below 75%;
- severe-risk proxy: visible instance area below 50%.

These thresholds operate on the whole instance mask. They do **not** locate the
class-defining defect.

## Status interpretation

- `INVALID_PAIRING`: A1/A2 do not isolate placement/visibility as intended.
- `REVIEW_LABEL_RISK`: pairing is valid, but at least one predeclared sensitive
  instance has less than 50% visible area in A2.
- `INTERNALLY_VALID`: pairing is valid and no predeclared sensitive instance
  crosses that severe proxy threshold.

None of these statuses means that a simulator is realistic. Realism and
utility on 300 g scenes remain `NOT_ESTABLISHED` until an identity-independent
real dense benchmark exists.

## Command

```bash
python -u -m coffee_detector.audit_vadcp_pair \
  --a1-root /content/sni21-vadcp-pilot/A1 \
  --a2-root /content/sni21-vadcp-pilot/A2 \
  --output /content/drive/MyDrive/coffee-bean-detection/sni21-vadcp-pilot-bundle/simulator_quality_report.json \
  --real-train-boxes 20959 \
  --real-median-bbox-area 0.012298
```

This command reads metadata only. It does not train a model, evaluate
validation/test images, or alter the datasets.

## Claim boundary

Permitted after a valid report:

- A1/A2 used the same observable selection and geometry plan;
- A2 changed the measured visibility/occlusion distribution;
- the declared sensitive classes have a quantified whole-instance visibility
  risk proxy.

Not permitted:

- the class-defining defect remained visible or became invisible;
- A2 is photorealistic;
- A2 improves detection on real dense scenes;
- the simulator represents 300 g by physical mass.
