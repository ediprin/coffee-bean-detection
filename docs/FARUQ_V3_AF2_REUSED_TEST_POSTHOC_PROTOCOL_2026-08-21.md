# Faruq-v3 AF2 Reused-Test Post-Hoc Protocol

Status: **frozen before AF2 test inference, by explicit user authorization**

## Scientific status

The Faruq-v3 test was previously consumed by the ACMC study. It is therefore
not an untouched locked test for AF2. This protocol deliberately reuses it as
a **post-hoc diagnostic**. No result may be described as a new locked-test or
independent-test confirmation.

## Fixed comparison

- Control: the three completed D0FT test reports from the ACMC study.
- Candidate: original AF2 checkpoints at seeds 42, 123, and 2026.
- Test package: the identical 129-parent, 208-instance Faruq-v3 package used by
  ACMC; all 21 classes are present, with rare-class support limitations.
- No training, threshold selection, checkpoint selection, or model tuning.
- Inference: image size 640, maximum 500 detections, confidence 0.001, IoU 0.7.
- Existing D0FT reports are reused only when their test-manifest hash matches.
- AF2 reports are cached by checkpoint and manifest SHA.

## Outcomes

Report paired three-seed Macro, Bottom-3, and Worst-class mAP50-95 plus the
1,000-iteration paired-parent bootstrap of the Macro delta. Bottom-3 and Worst
remain descriptive because class support is only 5--15 instances and 4--13
parents.

This post-hoc evaluation has no confirmatory PASS/FAIL gate. Its status is:

- `POSTHOC_DIRECTION_POSITIVE` when mean Macro delta is positive and AF2
  improves at least two of three seeds;
- otherwise `POSTHOC_DIRECTION_NOT_POSITIVE`.

The bootstrap interval and positive probability quantify uncertainty but do
not retroactively create a significance threshold. Whatever the result, no
further tuning, alternative checkpoint selection, or repeated test evaluation
is authorized.
