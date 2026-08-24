# Faruq-v3 AF2 Efficiency Audit Protocol

Status: **frozen before measurement**

## Question

What deployment cost does original AF2 add to the completed D0FT detector when
both models are measured on the same GPU and software runtime?

This audit is descriptive. It cannot change the completed model-selection or
mechanism conclusions, and it does not define a new PASS/FAIL accuracy gate.

## Fixed comparison

- Models: completed `D0FT` and original `AF2`.
- Seeds: 42, 123, and 2026, paired by seed.
- Training: none.
- Dataset inference: none; neither validation nor test images are required.
- Device: one CUDA GPU, whose exact name and compute capability are recorded.
- Precision: FP32.
- Input: one deterministic `1 x 3 x 640 x 640` CUDA tensor.
- Scope: synchronized model tensor-forward latency. Host-to-device transfer and
  detection postprocessing are excluded. The AF2 FFT frontend is included.
- Warmup: 30 iterations per checkpoint.
- Measurement: 100 iterations per checkpoint.
- Pair order alternates across seeds to reduce fixed order bias.

Each completed seed report contains both D0FT and AF2 measurements from the
same process and unchanged environment. An interrupted incomplete pair is
repeated. A completed SHA-matched pair may be reused after disconnection.

## Frozen metrics

For both models and all seeds, report:

1. total and trainable parameter counts;
2. parameter, buffer, state-tensor, and checkpoint-file bytes;
3. mean, median, and p95 tensor-forward latency;
4. throughput derived from mean latency;
5. resident, peak allocated, peak reserved, and incremental inference CUDA
   memory.

The summary reports paired deltas and ratios plus mean and population standard
deviation across seeds. Equal detector parameter count is required to support
the statement that AF2 is a parameter-free frontend. Checkpoint-file size is
reported separately and is not treated as a parameter-count proxy.

Standard YOLO FLOPs are deliberately omitted because ordinary module profilers
do not count the FFT frontend. The latency result is valid only for the
recorded GPU/software stack and the frozen tensor-forward scope; it is not an
end-to-end camera pipeline benchmark.

No result from this audit authorizes training, model tuning, or test access.
