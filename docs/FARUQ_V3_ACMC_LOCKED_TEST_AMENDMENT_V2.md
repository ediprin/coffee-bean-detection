# Faruq-v3 ACMC Locked-Test Amendment v2

Status: frozen after the v1 support audit FAIL and before any model inference,
2026-08-11.

## Reason for amendment

The v1 integrity audit opened labels but did not run a model. It established:

- 129 independent parent images;
- all 21 classes present;
- zero parent/hash overlap with development;
- one image per test parent and zero quarantined selected images;
- 5--15 instances and 4--13 independent parents per class.

The v1 test gate failed only because it required at least 10 instances and 5
parents for every class. No model output was available when this amendment was
written. The v1 FAIL remains part of the record and is not overwritten.

## Amended support gate

Inference is allowed only when the saved v1 audit also satisfies:

- at least 100 independent test images;
- at least 5 instances per class;
- at least 4 independent parents per class;
- every identity, coverage, and geometry safety gate from v1 remains PASS.

These thresholds match the observed floor only to establish feasibility. They
do not make rare-class AP precise. Consequently, bottom-three and worst-class
AP are descriptive secondary results and cannot reject or confirm ACMC.

## Primary endpoint and uncertainty

The primary endpoint is the mean paired ACMC1-minus-D0FT macro mAP50-95 delta
over seeds 42, 123, and 2026. Test confirmation requires:

- positive mean macro delta;
- positive macro delta on at least two of three seeds;
- at least 0.95 paired-parent-bootstrap probability that the mean delta is
  positive.

The bootstrap resamples the same 129 parent images for both heads and all three
seeds, with 1,000 iterations and seed 20260809. The runner also reports a 95%
percentile interval. Full all-class AP, bottom-three AP, worst-class AP, and
per-class AP remain visible.

## Finality

The six checkpoint hashes, test manifest hash, v1 audit hash, validation
confirmation hash, and this amendment hash are stored in the final report.
No training, architecture change, threshold tuning, or second test attempt is
authorized after inference, regardless of outcome.
