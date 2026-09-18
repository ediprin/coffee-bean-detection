# DefectosCafeVerde Physical-Bean Grouped Rebuild Result

Date completed: 2026-09-17

## Outcome

**PASS_GROUPED_DATASET_GATE**

The 4,038 original Roboflow project images and their original polygon
annotations were retrieved through the read-only API. Generated version-7
augmentations were not used. Sequential dual-sided views were assigned as one
atomic group before creating the split.

No detector training or model test evaluation was executed.

## Dataset summary

| Metric | Result |
|---|---:|
| Original images | 4,038 |
| Inferred physical groups | 2,069 |
| Two-view groups | 1,969 |
| One-view groups | 100 |
| Classes | 12 |
| Train images / groups | 2,827 / 1,448 |
| Validation images / groups | 808 / 408 |
| Test images / groups | 403 / 213 |
| Cross-split inferred physical groups | 0 |
| Exact cross-split image duplicates | 0 |
| Invalid image or label records | 0 |
| Augmentation included | No |

The achieved image proportions are 70.01% train, 20.01% validation, and
9.98% test. Every class occurs in every split.

## Class-instance distribution

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| agrio | 230 | 67 | 36 |
| broca | 166 | 48 | 25 |
| caracolillo | 360 | 104 | 58 |
| concha | 221 | 64 | 35 |
| elefante | 234 | 67 | 37 |
| helado | 150 | 42 | 22 |
| negro | 458 | 152 | 78 |
| normal | 319 | 94 | 50 |
| oreja | 151 | 44 | 23 |
| partido | 386 | 131 | 67 |
| seca | 437 | 132 | 83 |
| triangulo | 246 | 70 | 37 |

## Visual identity review

A 28-pair sample covering every one of the 14 filename prefixes was reviewed
side by side. The sequential even/odd pairs consistently show the same or
paired acquisition subject: matching bean shape and scale for single-bean
frames, and matching scene composition for the multi-bean `BL`/`BR` frames.
No sampled pair contradicted the grouping rule.

This supports the rule but does not turn it into author-provided ground truth.
The thesis must state that physical identity is inferred from (1) the paper's
dual-sided acquisition description, (2) the exact consecutive filename
structure, and (3) the visual review. Author confirmation would remain the
strongest provenance evidence.

## Frozen artifact checksums

| Artifact | SHA256 |
|---|---|
| `original_source_manifest.json` | `4dc14627076223c5529c9ccc8f9bff0eae00ebec3fb8581d4ad954b07cd9f384` |
| `grouped_manifest.json` | `66c03e8c3c23c82a66900a89b169a0d1a3052f591b1ca72ba4df13475f325d87` |
| `grouped_summary.json` | `469438fb143f4b096dde2979378637683b8eca90c24ef7bbf8cf3d7ff69c2a2a` |
| `grouped_audit.json` | `1b8aa1163d1690cbc6641329f8f427b0a197df142133b24ab6bd01899e5cca8b` |
| `pair_review_montage.jpg` | `a6a90b58d22227db505d91b9a44e22cd1ea762947d4c600b288a915a8132e3bd` |
| `defectoscafeverde-grouped-physical-v1.tar` | `53fb2233f1f0d1c77cb24eca2d720f86e0a16835b8a69f4e8f3176fae1aacef2` |

The portable TAR archive is 686,474,752 bytes and contains the complete
grouped dataset, manifests, YAML, and audit report.

## Decision boundary

The rebuilt dataset is suitable for a future frozen training protocol, subject
to the explicit inferred-identity limitation. This result does **not** by
itself authorize training, hyperparameter selection, or opening the rebuilt
test split. The next step is to freeze the model-comparison protocol using
train and validation only.
