# DefectosCafeVerde Physical-Bean Grouped Rebuild Protocol

Date frozen: 2026-09-17

## Purpose

Rebuild the paper-backed DefectosCafeVerde dataset without allowing the two
photographed sides of one physical bean to occur in different splits. This is
a dataset-preparation and audit protocol. It does not authorize training.

## Frozen source

- Project: `redtraining/defectoscafeverde`
- Source population: 4,038 original Roboflow project images
- Source retrieval: read-only Roboflow image-detail API and each record's
  `source.roboflow.com/.../original.jpg`
- Annotation source: original polygon annotations returned by the same image
  detail record; detection labels use their axis-aligned bounding boxes
- License reported by the project: CC BY 4.0
- Associated paper: *Dual-Sided Green Coffee Bean Defect Inspection Using a
  Mechatronic System with AI-Powered Computer Vision*

Generated version 7 is not used as the rebuild source because its training
partition contains three augmented derivatives per source image.

## Identity rule

The paper states that both sides of every bean are photographed. Source names
are an alphabetic class/acquisition prefix followed by a sequence number. The
frozen inferred physical identity is:

`lowercase(prefix) + floor(sequence_number / 2)`

Thus `A0` and `A1` remain together, as do `A2` and `A3`. This rule produces
2,069 candidate physical identities: 1,969 two-view groups and 100 one-view
groups. The identity is explicitly marked *inferred*, because the public
metadata does not expose a dedicated physical-bean ID.

## Split contract

- Seed: 42
- Ratios by original images: train 70%, validation 20%, test 10%
- Complete physical groups are assigned atomically.
- Assignment balances total images and the 12 class-instance distributions.
- Every class must occur in every split.
- Cross-split inferred physical groups must equal zero.
- No generated augmentation is retained. Runtime augmentation, if later
  authorized, may be applied to train only.
- Test is not read for model selection.

## Required post-build gates

1. Exactly 4,038 source images and 2,069 inferred physical groups.
2. Exactly 12 ontology classes, all present in every split.
3. Valid images and finite in-range YOLO labels.
4. Zero exact-image cross-split duplicates.
5. Zero inferred physical groups crossing splits.
6. Visual review of a stratified sample of even/odd pairs to validate the
   inferred identity rule.
7. A frozen SHA manifest for all source images and the resulting split.

Until these gates pass, `training_authorized` remains `false`.
