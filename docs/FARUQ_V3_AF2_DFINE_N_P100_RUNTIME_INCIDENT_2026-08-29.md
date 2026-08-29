# AF2 × D-FINE-N — Kaggle P100 runtime incident (2026-08-29)

## Status

`INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_RESULT`

The second seed-42 attempt did not reach detector training. `DFN0` failed during D-FINE model profiling because the Kaggle-allocated Tesla P100 has compute capability `sm_60`, while the preinstalled PyTorch build advertised only `sm_70` and newer CUDA architectures. The resulting error was `CUDA error: no kernel image is available for execution on the device`.

This failure is independent of AF2: the control arm failed before the paired treatment comparison was executed.

## Corrective action

The Kaggle notebook now freezes the runtime before cloning or training:

- `torch==2.5.1+cu124`
- `torchvision==0.20.1`
- official PyTorch CUDA 12.4 wheel index

The notebook then checks:

1. CUDA is available.
2. The allocated GPU compute capability is present in `torch.cuda.get_arch_list()`.
3. A real CUDA matrix operation executes and synchronizes successfully.
4. The v3 runner repeats the runtime contract and records it in static preflight provenance.

PyTorch 2.5.1+cu124 is used because its official binary includes `sm_60`, allowing the same frozen environment to support Kaggle P100 and newer T4-class GPUs.

## Scientific contract

No scientific setting changed: dataset split, seed, D-FINE commit/checkpoint, AF2 operator, optimizer/schedule, metrics, promotion thresholds, and locked-test policy remain unchanged.

The previous failed state ZIP must not be resumed. The next valid attempt starts clean from the immutable development archive.
