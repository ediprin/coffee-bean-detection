# Coffee Standard J25 Safe-Policy Root-Cause Result

Date: 2026-09-19

Status: **completed; validation only; test not opened**

All three models localized and matched all six validation instances of `Biji
Hitam Pecah`. `D0DIRECT` produced 2/6 correct raw class decisions, whereas
both `SAFED0` and `AF2LUMSAFE` at lambda zero produced 0/6. For `SAFED0`, the
six raw decisions were split between `Biji Pecah` and `Kulit Tanduk Ukuran
Besar`, three each. The target co-occurs in train images with `Biji Hitam
Penuh` twice, `Biji Hitam Sebagian` twice, and `Biji Cokelat` once.

The frozen sampler materially changes target exposure by more than 5%. Since
the raw correct-class count degraded under the combined safe policy, the next
single causal control removes the sampler while keeping semantic-safe
augmentation unchanged. This prioritization is diagnostic rather than causal
proof.

