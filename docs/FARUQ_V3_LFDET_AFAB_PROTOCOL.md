# Faruq-v3 LFDet AFAB Breadth Screening Protocol

## Purpose

Screen the raw-data frequency branch of Xu et al. (Neural Networks, 2025) on the controlled 21-class coffee benchmark without silently combining it with CGFI or FTIF.

Arms:

- **AF1**: AFAB-1 adaptive patch high-pass only.
- **AF2**: AFAB-2 entropy-conditioned directional amplitude suppression only.
- **AF12**: AFAB-1 followed by AFAB-2.

This is seed-42 breadth discovery, not confirmation.

## Paper-derived operator

LFDet first partitions the input image into overlapping patches and computes patch-wise DFT (PW-DFT). The paper uses patch size `m=32`.

### AFAB-1

For patch `i`, the paper defines patch energy

`E_i = sum |F_i|^2`

and patch-specific cutoff radius

`r_i = r_b * exp(1 - E_i / max(E))`,

where `r_b = (m/2) * r` and the reported radius ratio is `r=0.05`. A circular high-pass mask suppresses the low-frequency region inside the patch-specific radius.

### AFAB-2

The amplitude spectrum is aggregated by frequency direction to obtain `D_i(theta)`, normalized to a directional probability distribution, and its entropy is

`H_i = -sum p_i(theta) log p_i(theta)`.

The adaptive threshold is

`t_i = gamma / (1 + exp(-H_i))`,

with `gamma=0.1`. Directions whose normalized density is no larger than the threshold are suppressed; remaining amplitudes are multiplied by their normalized directional density. The original phase is retained before inverse DFT.

### Spatial gate

The paper describes min-max normalization of the recovered spatial-domain signal, element-wise multiplication with the raw image, and a residual operation. This implementation therefore uses

`I_out = I_raw + I_raw * MinMax(I_recovered)`.

AFAB is an input-time inference component in LFDet, not a train-only augmentation. The paper's speed analysis identifies AFAB, especially dense patch overlap, as a substantial inference-time cost.

## Paper ablation settings retained

- patch size: 32
- AFAB-1 radius ratio: 0.05
- AFAB-2 gamma: 0.1
- AF1, AF2 and AF12 are screened separately because the paper ablates the two filters separately; the full combination is also evaluated.

## Explicit transfer choices

The available paper text does not specify several implementation-level details. They are frozen here rather than presented as paper facts:

1. **Overlap 0.50 for breadth discovery.** LFDet explicitly studies 0.50 and 0.75. The denser 0.75 overlap improves continuity but costs speed. We use the paper-tested 0.50 setting during wide search to make three 50-epoch arms tractable. If an AFAB arm is retained, it must later be confirmed with overlap 0.75 before any claim about the paper-default/full setting.
2. **RGB treatment:** each RGB channel is frequency-filtered independently.
3. **Angular discretization:** the paper writes `theta in [0,360 degrees)` but does not give a code-level bin count. We freeze 360 bins and map each shifted-FFT pixel to `floor(theta_degrees)`.
4. **Overlap reconstruction:** processed patches are reconstructed by fold with overlap averaging.
5. **Numerical precision:** FFT/iFFT runs in float32 even under AMP, then returns to the input dtype, avoiding CUDA half-precision FFT restrictions.
6. **AF12 order:** AFAB-1 mask is applied before AFAB-2 directional weighting on the same patch spectrum.

These choices make this a controlled LFDet-AFAB transfer, not a claim of bit-for-bit reproduction.

## Computational implementation

At 640x640, patch 32 and overlap 0.50 give stride 16 and 39x39 = 1521 patches per image. To prevent a batch-wide complex FFT tensor from consuming excessive VRAM, processing is per image and patch FFTs are chunked. Unfolded and recovered patch columns are retained only for one image at a time, then reconstructed by `fold`.

AFAB-1's global maximum patch energy is computed per image/channel before filtering. With orthonormal FFT, Parseval allows this energy to be computed in the spatial patch domain without retaining every complex spectrum.

## Detection model contract

AFAB changes only the input tensor. The YOLO26 architecture and parameter state remain native:

- D0 detector weights are transferred bitwise;
- AFAB has no learnable parameters or persistent state;
- the same AFAB operator is active for training tensor forwards and evaluation/inference tensor forwards;
- box and classification heads are otherwise untouched.

Unlike classification-only modules such as ACMC/CGFI, AFAB can affect both localization and classification indirectly because it changes the pixels entering the entire detector. This is part of the AFAB hypothesis and must not be described as box-branch preservation.

## Frozen breadth settings

- seed: 42
- epochs: 50
- image size: 640
- batch: 16
- patch: 32
- overlap: 0.50
- radius ratio: 0.05
- gamma: 0.1
- angular bins: 360 (transfer choice)
- evaluation: validation only
- locked test: unavailable/forbidden

## Discovery decision

Each arm is compared with D0FT using the shared breadth-search rule:

- macro mAP50-95 no worse than D0FT by >0.2 pp;
- bottom-3 no worse by >2 pp;
- worst-class no worse by >3 pp;
- at least one signal: macro +0.2 pp, bottom-3 +0.5 pp, or worst +0.5 pp.

A `RETAIN` result authorizes later confirmation only. It is not final evidence.
