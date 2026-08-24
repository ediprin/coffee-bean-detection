# Faruq-v3 AF2 Same-Device Efficiency Audit Result

Date: 2026-08-21

Protocol: `docs/FARUQ_V3_AF2_EFFICIENCY_AUDIT_PROTOCOL_2026-08-21.md`

Models: completed D0FT and original AF2 checkpoints, paired at seeds
42/123/2026

Training executed: **no**

Dataset/test accessed: **no**

## Validity

All frozen validity gates passed:

- the environment was unchanged inside every D0FT--AF2 pair;
- detector parameter counts were equal;
- every report recorded `training_executed=false`;
- every report recorded `test_images_accessed=false`.

All three pairs used a Tesla T4 (compute capability 7.5), PyTorch
2.11.0+cu128, CUDA 12.8, and Ultralytics 8.4.96. Measurements used FP32,
batch 1, 640 x 640 input, 30 warmup iterations, and 100 synchronized
tensor-forward measurements per checkpoint. Host-to-device transfer and
postprocessing were excluded; the AF2 FFT frontend was included.

## Headline result

| Metric | D0FT mean | AF2 mean | AF2 - D0FT | AF2 / D0FT |
|---|---:|---:|---:|---:|
| Parameters | 2,511,990 | 2,511,990 | 0 | 1.000x |
| State-tensor bytes | 10,124,840 | 10,124,840 | 0 | 1.000x |
| Checkpoint-file bytes | 5,389,786 | 5,401,040 | +11,254 | 1.002x |
| Mean latency | 14.54 ms | 25.35 ms | +10.81 ms | 1.748x |
| Median latency | 13.52 ms | 23.59 ms | +10.07 ms | 1.745x |
| p95 latency | 19.15 ms | 33.78 ms | +14.63 ms | 1.767x |
| Throughput | 68.93 image/s | 39.96 image/s | -28.98 image/s | 0.581x |
| Peak allocated CUDA memory | 75,220,480 B | 127,555,072 B | +52,334,592 B | 1.696x |
| Incremental inference peak | 26,345,472 B | 78,667,776 B | +52,322,304 B | 2.986x |

The checkpoint-file increase was only 0.21%, while median tensor-forward
latency increased by 74.5%, p95 latency by 76.7%, and peak allocated memory by
69.6%. Mean throughput retained 58.1% of D0FT throughput.

AF2 also carried 12,288 additional non-persistent buffer bytes. These buffers
did not increase the detector parameter count or serialized state-tensor
bytes, so the parameter-free frontend statement is supported. It must not be
misread as compute-free or memory-free.

## Paired latency stability

| Seed | D0FT median | AF2 median | Ratio | Measurement order |
|---:|---:|---:|---:|---|
| 42 | 13.54 ms | 21.25 ms | 1.570x | D0FT -> AF2 |
| 123 | 13.54 ms | 21.20 ms | 1.566x | AF2 -> D0FT |
| 2026 | 13.49 ms | 28.32 ms | 2.099x | D0FT -> AF2 |

The direction is consistent in every pair, but AF2 latency varied more than
D0FT latency. The reported three-seed mean and ratio therefore retain the
observed runtime variability rather than presenting one favorable run.

## Accuracy--efficiency synthesis

The completed paired accuracy confirmation reported:

| Validation metric | D0FT mean | AF2 mean | AF2 gain |
|---|---:|---:|---:|
| Macro mAP50-95 | 86.62% | 87.94% | +1.32 points |
| Bottom-3 mAP50-95 | 76.58% | 79.37% | +2.80 points |
| Worst-class mAP50-95 | 73.05% | 78.15% | +5.10 points |

Together with the mechanism diagnostic, the evidence supports the following
bounded statement:

> AF2 is a parameter-free frequency-angular input frontend that improves
> fine-grained classification/ranking and disproportionately benefits the
> lower tail, while adding substantial FFT latency and temporary CUDA memory.

AF2 remains the selected research model because the largest accuracy gains
occur in Bottom-3 and Worst-class performance, which are central to the
fine-grained defect problem. D0FT remains the preferable deployment option
when throughput or memory has priority over lower-tail accuracy.

## Limits

- This is a synchronized GPU tensor-forward benchmark, not full camera-system
  latency or end-to-end FPS.
- The approximately 40 image/s AF2 throughput excludes image decode,
  host-to-device transfer, detection postprocessing, and I/O.
- Standard YOLO FLOPs are not reported because ordinary profilers omit the FFT
  frontend.
- `model_sample_resident_bytes` was allocator-order sensitive in the first
  pair and is not used for a scientific claim. Peak allocated and incremental
  inference peak were identical across all three pairs and are the retained
  memory measures.
- Runtime values apply to the recorded T4/software stack and should not be
  generalized unchanged to other hardware.

Machine-readable evidence:
`docs/evidence/FARUQ_V3_AF2_EFFICIENCY_AUDIT_2026-08-21.json`.

Source artifact:
`experiments/faruq-v3-af2-efficiency-audit-v1/af2_efficiency_summary.json`.
