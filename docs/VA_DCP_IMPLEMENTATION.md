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
| A1 | real + empirical-layout copy-paste | YOLO26n |
| A2 | A1 + visibility-aware 2.5D layers | YOLO26n |

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

The command estimates the foreground against the border background, selects the
largest component for classification images or the component nearest the
annotated box centre for YOLO crops, fills holes, crops the bean, saves an RGBA
asset, removes pixel-identical assets, and records failures in
`object_library.json`.
By default it retains at most 500 successful assets per class, sampled with seed
42, so a dataset containing hundreds of thousands of boxes is not exhaustively
cropped merely for a pilot.

YOLO object-library indexing uses a class-balanced reservoir sampler. It parses
labels without opening, hashing, or calculating perceptual features for every
image, caps candidates from one source image, and only opens/hashes selected
assets. This avoids the former full-dataset audit cost during cutout sampling.

The source bean must be complete and unoccluded; otherwise its mask is not a
valid `full_mask`. Assets whose foreground touches the crop boundary are rejected,
but visual review is still required to remove partially hidden source beans.

Do not continue before visually checking masks from every class.  Automatic
foreground extraction is a draft annotation, not unquestionable ground truth.

## 2. Calibrate the scene prior from real train

Count, projected scale, within-frame scale variation, and background statistics
are learned only from the real train split. One scene-scale is sampled first and
all beans in that frame receive only the empirical within-scene residual. This
avoids mixing camera distances inside one synthetic frame.

```bash
python -u -m coffee_detector.profile_vadcp_source \
  --data-root /path/to/real-dataset \
  --output /path/to/real-scene-calibration.json \
  --seed 42
```

The profile stores a compact empirical quantile sample, not training images.
It must be rebuilt when the real train split or camera geometry changes.

## 3. Generate A1 and A2

Use the same real dataset, object library, number of generated scenes, and seed.
Only `mode` differs.

```bash
python -u -m coffee_detector.generate_vadcp_dataset \
  --real-data-root /path/to/real-dataset \
  --object-library /path/to/object-library \
  --background-root /path/to/blank-real-backgrounds \
  --scene-profile /path/to/real-scene-calibration.json \
  --output-root /path/to/A1-naive \
  --mode naive \
  --synthetic-images 2000 \
  --seed 42

python -u -m coffee_detector.generate_vadcp_dataset \
  --real-data-root /path/to/real-dataset \
  --object-library /path/to/object-library \
  --background-root /path/to/blank-real-backgrounds \
  --scene-profile /path/to/real-scene-calibration.json \
  --output-root /path/to/A2-vadcp \
  --mode visibility \
  --synthetic-images 2000 \
  --seed 42
```

If `--background-root` is omitted, the generator creates a procedural light
background whose color, low-frequency gradient, and sensor-noise range are
calibrated from unannotated pixels in real train. This remains a fallback for a
software smoke test. The final thesis experiment should use blank tray/conveyor
frames captured by the target camera.

The implementation is deliberately described as **physics-informed 2.5D
projected packing**, not as a 3D rigid-body simulator:

- `spread`, `cluster`, and `pile` scenes use contact/overlap constraints;
- one z-order creates full and visible masks;
- 25% of instances are visibility-controlled in A2;
- severe/extreme cases use multiple empirical-size occluders;
- one light direction, exposure, white balance, camera noise, and scale-aware
  contact/cast-shadow model is shared by the whole frame;
- the empirical signed width/height distribution selects each rotation, so
  diagonal rotation does not collapse axis-aligned bean boxes toward squares;
- target aspect ratios are sampled per class and assigned hardest-first to
  real cutouts whose intrinsic silhouette can realize them; pixels are never
  stretched anisotropically, and every unavoidable fallback is counted;
- YOLO boxes from non-square images are converted to an isotropic pixel frame
  before scale, distance, area, and aspect statistics are calibrated or audited;
- an older object-library manifest is geometry-profiled once with visible
  progress, saved atomically, and then reused by both paired augmentation arms;
- uncertain mask edges are eroded into a trusted color core, feathered inward,
  edge-bled, and transformed in premultiplied-alpha space to suppress paste
  halos without expanding annotation masks;
- the same scene seed, objects, scales, rotations, background, and photometry
  are preselected for A1/A2; only visibility-aware placement differs.

Every synthetic instance stores:

```text
class_id, source_asset_id, source_id, source_split,
z_order, visible_bbox, full_bbox,
visible_mask, full_mask, visibility_ratio, visibility_bin
```

Only visible boxes with visibility at least 0.10 enter the YOLO label.  Full
masks remain metadata and are not presented as visible supervision.

## 4. Audit generated data

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

The visual runner also writes `contact_sheet_raw.jpg`. Audit cutout boundaries
independently on black, white, and checkerboard backgrounds:

```bash
python -u -m coffee_detector.run_cutout_visual_audit \
  --object-library /path/to/object-library \
  --output-root /path/to/cutout-audit \
  --samples 18
```

Internal correctness is not evidence of realism. Compare visible YOLO geometry
against the real train distribution. The report uses matched-size real-vs-real
resampling as its null variation rather than a universal arbitrary distance:

```bash
python -u -m coffee_detector.audit_vadcp_realism \
  --real-data-root /path/to/real-dataset \
  --synthetic-data-root /path/to/A2-vadcp \
  --output /path/to/A2-realism.json \
  --seed 42
```

It reports visible labeled density separately from generated density, plus
long-side, area, aspect ratio, nearest-neighbor spacing, overlap, border touch,
and synthetic visibility/ignored rates. `PASS_GEOMETRY` still requires manual
approval of the raw and cutout contact sheets.

## 5. Run the locked ablation

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

## 6. Visibility-stratified real-test evaluation

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

## Method basis

The design follows four results rather than inventing unconstrained graphics:

- Toda et al. generated dense barley-seed images from cutouts, controlled
  overlap, and automatically derived instance labels—the closest crop/seed
  precedent: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7160130/>.
- Simple Copy-Paste shows that correct mask/box updates and a strong paired
  baseline matter more than elaborate blending:
  <https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html>.
- CrowdAug shows targeted crowded placement and paste order provide useful
  pseudo-depth for crowded detection:
  <https://ojs.aaai.org/index.php/AAAI/article/view/25124>.
- Coffee inspection literature already stratifies synthetic scene density and
  rejects implausible overlap, supporting a domain-specific quality gate:
  <https://www.mdpi.com/2076-3417/9/19/4166>.
