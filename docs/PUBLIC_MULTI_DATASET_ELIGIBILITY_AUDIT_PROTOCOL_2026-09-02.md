# Public Multi-Dataset Eligibility Audit Protocol

Status: **FROZEN AUDIT PROTOCOL — NO TRAINING AUTHORIZATION**  
Date: 2026-09-02

## Question

Can at least three genuinely independent public coffee-bean object-detection
dataset lineages support the proposal's V2 multi-dataset design?

This audit precedes model selection. It must not train a detector, evaluate a
checkpoint, tune AF2, or inspect model performance on any split.

## Frozen inputs

The candidate metadata is stored in
`configs/public_dataset_audit/v2_candidate_registry.yaml`. Each acquired export
must be bound to an explicit owner, project, version, license, archive SHA256,
and extracted YOLO root. Project overview counts are not accepted as a version.

## Automated checks

For each dataset the runner records:

- decodable image and valid YOLO/polygon label counts;
- image, box, class, split, empty-image, source-parent, and resolution counts;
- missing classes, placeholder/ambiguous classes, and invalid annotations;
- exact-image and Roboflow-parent leakage across splits;
- dHash/mean-color near-duplicate candidates across splits;
- archive existence and exact SHA256 agreement.

Across datasets it records exact-image overlap, perceptual-match candidates,
and exact-hash lineage components. Perceptual matches are review candidates,
not automatic proof of shared lineage.

## Per-dataset status

- `PASS_AS_IS`: metadata and archive verified, valid annotations, complete
  split, and no detected split leakage.
- `REBUILD_GROUPED_SPLIT`: usable originals are available, but the official
  split is missing or leaks exact/Roboflow-parent identities.
- `REVIEW_NEAR_DUPLICATES`: no definitive leakage was found, but perceptual
  candidates cross splits.
- `HOLD_METADATA`: version/license/archive hash is not fully verified.
- `REJECT`: corrupt/invalid data, empty declared classes, fewer than 100
  estimated source parents, or unresolved placeholder classes.
- `NOT_ACQUIRED`: the frozen export is not locally available.

`REBUILD_GROUPED_SPLIT` does not authorize training. It only means a new split
may be constructed after review from source/duplicate clusters.

## V2 gate

At least three eligible exact-hash lineage components must remain. Any
unreviewed perceptual overlap blocks the gate. Even `PASS_V2_DATASET_GATE`
keeps `training_authorized=false`; grouped split manifests, visual annotation
review, ontology decisions, development-dataset selection, and the experiment
protocol must still be frozen separately.

## Command

```bash
python -m coffee_detector.analysis.public_dataset_eligibility \
  --registry configs/public_dataset_audit/v2_candidate_registry.yaml \
  --output-root evidence/public-dataset-v2-audit \
  --dataset-root lulus_v1=/path/to/lulus-v1 \
  --archive lulus_v1=/path/to/lulus-v1-yolov8.zip
```

Repeat `--dataset-root CODE=PATH` and `--archive CODE=PATH` for each acquired
candidate. Outputs are a complete JSON summary, one JSON per dataset, a concise
Markdown report, and CSV review queues for exact and perceptual cross-dataset
matches.
