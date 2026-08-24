# SNI-21 Full-Frame Context Paired Analysis

## Status

This is a post-hoc exploratory diagnostic. It does not replace the official
Ultralytics mAP result and is not a pre-registered confirmatory test.

## Motivation

The original diagnostic assigned a class using the localized prediction with
the highest IoU. At confidence 0.001, YOLO can retain several class-specific
candidates at almost the same coordinates. Selecting by IoU can therefore use
a lower-confidence class and bias conditional class accuracy. Official AP is
unaffected because it is computed by Ultralytics.

## Corrected decision rule

For every one-object image and arm:

1. collect predictions whose IoU with the ground-truth box is at least 0.5;
2. if none exists, record `miss`;
3. otherwise choose the candidate with highest confidence, breaking ties by
   IoU;
4. compare its class with the ground-truth class.

FC1 and FC2 must contain identical image identities. Analysis reads their
existing `prediction_records.jsonl`; it performs no training, inference, or
test access.

## Statistics

- Report micro and class-macro top-1 accuracy and proposal recall.
- Report per-class deltas and correct/wrong/miss transitions.
- Use 10,000 class-stratified paired bootstrap iterations for FC2 minus FC1.
- Use an exact two-sided McNemar test on top-1 correct versus incorrect.

Exploratory background harm is marked supported only when official mAP falls,
the 95% bootstrap interval for macro top-1 delta lies below zero, and McNemar
has p below 0.05. This conclusion is restricted to the development subset.
