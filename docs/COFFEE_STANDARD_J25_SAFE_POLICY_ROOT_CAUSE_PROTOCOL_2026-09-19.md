# Coffee Standard J25 Safe-Policy Root-Cause Audit

Status: **frozen before diagnostic evaluation**

Date: 2026-09-19

This no-training audit determines which single component ablation should
follow `SAFED0`. It uses only the unchanged train metadata and validation
images; the locked test is not extracted or evaluated.

The audit reports for `Biji Hitam Pecah`:

- train images, instances, source identities, and sibling multiplicity;
- natural image share and expected share under the frozen repeat sampler;
- class repeat factor and co-occurring classes;
- raw top-500 and low-confidence final localization/classification counts for
  `D0DIRECT`, `SAFED0`, and `AF2LUMSAFE` at inference strength zero;
- complete wrong-class destinations.

If the safe policy reduces raw correct-class decisions while the sampler
materially changes target exposure, the next priority is safe augmentation
without the sampler. If sampler exposure is nearly unchanged, the next
priority is the sampler with standard augmentation. This is prioritization,
not causal proof; no training is authorized by this diagnostic protocol.

