# Coffee Standard J25 CWCF2 Gate Audit — 2026-09-20

Status: **validation-only diagnostic frozen before execution**.

The first CWCF2 run is quarantined because multiple runtimes wrote the same
training directory. Its checkpoint cannot support a performance claim, but it
can be used to decide whether a clean rerun is worth the compute.

The audit evaluates the quarantined checkpoint twice on the unchanged J25
train-siblings validation split:

1. the checkpoint as saved, with learned explicit-composition gates active;
2. the same checkpoint with only the three composition gates set to zero.

No weight is optimized. The zero-gate checkpoint is temporary and local to the
runtime. The test split remains closed. Active metrics must reproduce the
quarantine endpoint before the ablation is accepted.

This comparison isolates the direct inference contribution of the explicit
attribute-to-class composition. It does not undo the auxiliary attribute loss
or any representation changes accumulated during training. Therefore:

- improvement after zeroing supports harmful inference-time composition;
- a negligible gate effect or a zero-gate endpoint still below CWCF1 supports
  training-path damage;
- neither outcome is a formal model result because the parent checkpoint is
  quarantined.

The default next action is to stop CWCF2 and retain CWCF1. A clean CWCF2 rerun
is not authorized by this diagnostic.
