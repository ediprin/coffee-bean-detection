# Coffee Standard J25 Primary-Candidate Protocol

Status: **FROZEN DATA PROTOCOL — TRAINING NOT YET AUTHORIZED**  
Date: 2026-09-18

## Purpose

This protocol determines whether the public dataset associated with Sayid
Muhammad Jundullah's thesis and the *Coffee Detection With Standard* Roboflow
project can replace the untraceable Faruq lineage as the primary development
dataset for fine-grained green-coffee defect detection.

This is a data decision, not a model-selection experiment. It does not inspect
AF2, baseline, validation, or test performance.

## Frozen source

- Thesis repository record: Universitas Malikussaleh, *YOLOv8-Based
  Multi-Class Detection of Coffee Bean Defects and Contaminants for Automated
  Quality Grading*.
- Public project: `tes-rcphs/coffee-detection-with-standard`.
- Frozen export: version 8, YOLOv8 format.
- Reported license: CC BY 4.0.
- Frozen archive SHA256:
  `5529de365ad888406b5534a5d1bf5a4a29c9937bc095b16174f8516d396bb1fc`.

The existing raw audit found 993 exported derivative images, 13,926 boxes, 25
labels, and 14 Roboflow-parent identities crossing the supplied splits. The
official split is therefore not used.

## Provenance discrepancy that must remain visible

The sources are related but not numerically interchangeable:

- the thesis/public project describes 451 images, 6,487 annotations, and 25
  SNI+ICO categories;
- the journal paper reports 2,000 augmented images, 3,983 labels, and 20
  categories;
- the frozen v8 export contains 993 images and 13,926 boxes;
- its filenames yield a recoverable parent-component count that does not equal
  the thesis image count.

Consequently, the rebuilt artifact is called **J25 primary candidate**, not
the paper's 20-class dataset and not canonical SNI-21. These discrepancies
must be disclosed and resolved through author confirmation or source metadata
before the artifact becomes the formal primary dataset.

## Ontology rule

All 25 source labels are preserved exactly and in their original order. No
class is renamed, merged, removed, or force-mapped during the primary-candidate
build.

The seven labels that were excluded from the earlier SNI-21 external benchmark
remain valid J25 labels here: moldy bean, silver-skin bean, sour bean, gravel,
and three twig-size labels. The artifact is described as SNI+ICO because a
canonical one-to-one SNI-21 equivalence has not been established.

## Identity and split rule

1. Images sharing a Roboflow parent identifier are one identity component.
2. Exact image hashes also join identity components.
3. Exactly one deterministic representative is retained per component. The
   representative with the most boxes is preferred; deterministic hashing
   resolves ties without model results.
4. Generated siblings are not counted as independent images and are excluded.
5. Components are assigned approximately 70/15/15 to train/validation/test by
   a seeded, class-aware optimizer.
6. No parent or exact hash may cross a split.
7. Runtime augmentation may later be applied to train only. Validation and test
   never use sibling augmentations.

## Technical gates

The build must satisfy all of the following:

- exact 25-label J25 ontology;
- one selected representative per identity component;
- zero parent overlap across splits;
- zero exact-hash overlap across splits;
- all 25 labels occur in train, validation, and test;
- each achieved split fraction is within three percentage points of its target;
- all images decode and all YOLO annotations are valid.

Failure means the dataset is not technically splittable under this protocol.

## Remaining authorization gates

Passing the technical gates still leaves `training_authorized=false`. Before
training, the following must be frozen:

1. visual review of a class-stratified sample from every split;
2. review of suspicious boxes and source-sibling disagreements;
3. provenance resolution for the 451/993/2,000-image and 25/20-class mismatch;
4. a final decision whether J25 or another paper-backed dataset is the
   development dataset;
5. baseline, AF2 conditions, seeds, metrics, and test-lock protocol.

If provenance cannot be resolved, J25 remains an external/provisional dataset
and must not be represented as the exact dataset evaluated in the journal
paper.

## Command

```bash
python -m coffee_detector.data.prepare_coffee_standard_primary \
  --source-root /content/coffee-standard-v8-raw \
  --output-root /content/coffee-standard-j25-grouped \
  --seed 42 \
  --link-mode auto
```

Primary artifacts:

- `coffee_standard_j25_manifest.json`;
- `coffee_standard_j25_components.json`;
- `coffee_standard_j25_summary.json`;
- `data.yaml` and grouped YOLO split directories.

