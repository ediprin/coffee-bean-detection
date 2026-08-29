# AF2 × D-FINE-N infrastructure fix — 2026-08-29

## Status

The first Kaggle attempt did **not** produce an experimental result. Training stopped in the control arm `DFN0` before a completed epoch, and `DFN_AF2` was not run.

The static pairing checks before training had passed: native/candidate parameter counts were equal, initialized detector state was matched, and AF2 had zero learned parameters.

## Root cause

The YOLO-to-COCO adapter wrote the 21 coffee categories as COCO IDs `1..21` while the generated D-FINE configuration used:

```yaml
num_classes: 21
remap_mscoco_category: false
```

At the pinned D-FINE commit, when `remap_mscoco_category` is false, annotation `category_id` values are used directly as target labels. Therefore a 21-class detector requires labels `0..20`.

The invalid target label `21` reached the Hungarian matcher, which indexes class probabilities by target label. CUDA reported a device-side `index out of bounds` assertion; because CUDA execution is asynchronous, the Python traceback surfaced later at `torch.cdist`, but the category-index mismatch is the causal infrastructure error.

## Fix

The Kaggle notebook now launches `scripts/run_dfine_af2_kaggle_screen_v2.py`.

The v2 launcher changes only the dataset-adapter contract:

- COCO category declarations are rewritten from `1..21` to `0..20`;
- annotation `category_id` values are rewritten from `1..21` to `0..20`;
- a new fail-closed preflight check verifies exact category IDs `0..20` and verifies all annotation labels are within `[0, 20]` for both train and validation manifests.

No AF2 parameter, detector architecture, dataset membership, initialization rule, optimizer/schedule, seed, metric, promotion gate, or test policy is changed.

## Scientific interpretation

The failed first attempt is classified as **INFRASTRUCTURE_FAILURE_NO_RESULT**. It provides no evidence for or against AF2 transfer to D-FINE-N and must not be included in model-performance comparisons.
