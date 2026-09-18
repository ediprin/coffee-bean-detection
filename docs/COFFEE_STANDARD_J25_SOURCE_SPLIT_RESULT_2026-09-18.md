# Coffee Standard J25 Source-Split Result

Date: 2026-09-18
Status: **PASS (dataset construction only; training not authorized by this audit)**

## Outcome

The thesis-author archive can be reconstructed into 451 source identities
without using filename prefixes alone. Source identity is defined by the
conjunction of the augmentation-stripped basename and exact label-file bytes.
This recovers exactly 271 groups of three training derivatives and 180
single-source images from the supplied validation and test partitions.

Within the recovered training triplets, the minimum normalized structural
similarity is 0.9843925, above the frozen 0.98 gate. A deterministic,
label-only grouped assignment at seed 42 produces a class-complete source-level
development split and a locked test manifest:

| Split | Source representatives | Instances | Class coverage |
|---|---:|---:|---:|
| Train | 315 | 4,297 | 25/25 |
| Validation | 68 | 1,164 | 25/25 |
| Locked test | 68 | 1,025 | 25/25 |

All construction gates passed:

- exactly 451 recovered source identities;
- exactly 271 three-member augmentation groups and 180 singleton groups;
- all training sibling groups contain exactly three derivatives;
- all 25 classes occur in train, validation, and locked test;
- locked-test images are not extracted into the development dataset;
- the generated development YAML contains no test path.

This result authorizes use of the reconstructed **train and validation** data
only under a separately frozen experiment protocol. It does not authorize
training by itself and does not open the locked test.

## Reproducibility

- Author archive: `data_aug_11.zip`
- Archive SHA256:
  `a4d8570e7de8d0ebba408d1d736256806e8f5ff1ebcf62cafd2fa13d149d613d`
- Builder: `coffee_detector.data.prepare_coffee_standard_j25_source_split`
- Summary artifact: `coffee_standard_j25_source_split_summary.json`
- Manifest artifact: `coffee_standard_j25_source_split_manifest.json`
- Model training executed: **false**
- Test images accessed by a model: **false**
