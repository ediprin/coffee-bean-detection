# VA-DCP implementation protocol

## Scope

`Visibility-Aware Dense Copy-Paste` (VA-DCP) is implemented as an offline data
pipeline.  It does not modify the Coffee-17 classifier and does not patch
Ultralytics internals.  The materialized output remains an ordinary YOLO
detection dataset, while richer visible/full-mask metadata is stored separately.

The first locked ablation is:

| Arm | Training data | Detector |
|---|---|---|
| A0 | real only | YOLO26n |
| A1 | real + naive copy-paste | YOLO26n |
| A2 | real + VA-DCP | YOLO26n |

Curriculum (A3) and visibility-weighted loss are deliberately deferred until A2
beats A1.  This prevents a failed data hypothesis from being hidden by several
simultaneous modifications.

## Data contract

The real dataset must have leakage-safe `train`, `val`, and `test` splits in the
usual YOLO layout.  Only `train` assets may enter the object library.  Synthetic
images are added to train; val and test are copied unchanged from the real
dataset.

```text
real dataset
├── train/images + train/labels
├── val/images + val/labels
├── test/images + test/labels
└── data.yaml
```

For final thesis claims, val/test must contain real dense scenes.  A synthetic
test set is not accepted as real-world validation.

## 1. Build an object library

From a YOLO detection dataset:

```bash
python -u -m coffee_detector.prepare_object_library \
  --data-root /path/to/real-dataset \
  --output-root /path/to/object-library \
  --source-split train
```

From a single-bean classification dataset:

```bash
python -u -m coffee_detector.prepare_object_library \
  --classification-root /path/to/coffee17 \
  --output-root /path/to/object-library \
  --allowed-splits train unspecified
```

The command estimates the foreground against the border background, keeps the
largest component, fills holes, crops the bean, saves an RGBA asset, removes
pixel-identical assets, and records failures in `object_library.json`.
By default it retains at most 500 successful assets per class, sampled with seed
42, so a dataset containing hundreds of thousands of boxes is not exhaustively
cropped merely for a pilot.

The source bean must be complete and unoccluded; otherwise its mask is not a
valid `full_mask`. Assets whose foreground touches the crop boundary are rejected,
but visual review is still required to remove partially hidden source beans.

Do not continue before visually checking masks from every class.  Automatic
foreground extraction is a draft annotation, not unquestionable ground truth.

## 2. Generate A1 and A2

Use the same real dataset, object library, number of generated scenes, and seed.
Only `mode` differs.

```bash
python -u -m coffee_detector.generate_vadcp_dataset \
  --real-data-root /path/to/real-dataset \
  --object-library /path/to/object-library \
  --background-root /path/to/blank-real-backgrounds \
  --output-root /path/to/A1-naive \
  --mode naive \
  --synthetic-images 2000 \
  --seed 42

python -u -m coffee_detector.generate_vadcp_dataset \
  --real-data-root /path/to/real-dataset \
  --object-library /path/to/object-library \
  --background-root /path/to/blank-real-backgrounds \
  --output-root /path/to/A2-vadcp \
  --mode visibility \
  --synthetic-images 2000 \
  --seed 42
```

If `--background-root` is omitted, the generator creates a procedural light
background.  That is suitable for a software smoke test, not the final thesis
experiment.

Every synthetic instance stores:

```text
class_id, source_asset_id, source_id, source_split,
z_order, visible_bbox, full_bbox,
visible_mask, full_mask, visibility_ratio, visibility_bin
```

Only visible boxes with visibility at least 0.10 enter the YOLO label.  Full
masks remain metadata and are not presented as visible supervision.

## 3. Audit generated data

```bash
python -u -m coffee_detector.audit_vadcp \
  --data-root /path/to/A1-naive

python -u -m coffee_detector.audit_vadcp \
  --data-root /path/to/A2-vadcp
```

The audit verifies:

- visible mask is a subset of full mask;
- stored visibility equals visible area divided by full area;
- z-order reproduces every visible mask;
- visible/full boxes match their masks;
- no val/test source asset enters synthetic train;
- YOLO labels match non-ignored instances;
- the ordinary dataset leakage audit also passes.

Create a contact sheet before training:

```bash
python -u -m coffee_detector.run_vadcp_visual_audit \
  --data-root /path/to/A2-vadcp \
  --output-root /path/to/A2-visual-audit \
  --samples 20
```

Blue boxes are visible boxes, yellow dashed boxes are full/amodal boxes, and a
pink box marks the visibility-controlled focus instance.

## 4. Run the locked ablation

Start with seed 42:

```bash
python -u -m coffee_detector.run_vadcp_ablation \
  --arm A0=/path/to/real-dataset \
  --arm A1=/path/to/A1-naive \
  --arm A2=/path/to/A2-vadcp \
  --seeds 42 \
  --output-root /path/to/vadcp-results \
  --device 0
```

The runner audits each arm, resumes interrupted runs, evaluates locked real test
data, and reports mAP plus count error.  It treats a run as complete only when
both `best.pt` and `experiment_manifest.json` exist.

## 5. Visibility-stratified real-test evaluation

The final real test metadata uses the same COCO-like fields `image_id`,
`category_id`, `bbox`, and `visibility_bin`. Objects from non-target bins are
handled as ignore regions during a bin-specific AP calculation.

```bash
python -u -m coffee_detector.evaluate_visibility \
  --checkpoint /path/to/A2_seed42/weights/best.pt \
  --data-root /path/to/real-dataset \
  --metadata /path/to/instances_real_test_visibility.json \
  --output /path/to/A2_seed42_visibility.json \
  --device 0
```

This evaluator reports AP50 and mAP50-95 for clear, mild, severe, and extreme,
plus count MAE, signed bias, exact-count rate, and duplicate-prediction rate.
The generator does not invent real-test visibility labels.  Those must be
annotated on genuinely dense real images.  When exact amodal area is uncertain,
two annotators should assign the four ordinal bins independently and agreement
should be reported.

Only after A2 gives a useful signal should it be confirmed with three seeds:

```bash
python -u -m coffee_detector.run_vadcp_ablation \
  --arm A0=/path/to/real-dataset \
  --arm A1=/path/to/A1-naive \
  --arm A2=/path/to/A2-vadcp \
  --seeds 42 123 2026 \
  --output-root /path/to/vadcp-results \
  --device 0
```

## Decision gate

A2 advances to curriculum only if it improves the real dense test set relative
to both A0 and A1, especially count MAE, count bias, recall, and difficult
visibility strata.  Overall mAP must not fall materially.  If it fails, do not
add a custom loss to rescue the claim; inspect mask quality, compositing realism,
and the real-test domain gap first.
