# Bab III Primary-Source Hardening — 2026-08-25

Status: **source-provenance audit for proposal methodology**.

Purpose: prevent the Bab III AF2 and YOLO26 descriptions from mixing (a) statements directly supported by the parent papers, (b) repository transfer/engineering decisions, and (c) the study's frozen experimental protocol.

This file is an audit artifact. It is not a substitute for the proposal prose.

---

## 1. Source hierarchy used for Bab III

For methodological claims, use the following authority order:

1. **primary method paper** for the method actually being described;
2. **repository implementation/config** for exact transfer and engineering decisions;
3. **frozen experiment protocol** for study-specific training, fairness, dataset, and evaluation rules;
4. historical result documents only for historical/pilot status, never to redefine the prospective method.

Canonical keys:

- `[FG-01]` — Xu et al. (2025), *More signals matter to detection: Integrating language knowledge and frequency representations for boosting fine-grained aircraft recognition*, Neural Networks 187, 107402.
- `[DET-01]` — Jocher et al. (2026), *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*, arXiv:2606.03748v1.

Repository anchors:

- `src/coffee_detector/afab/operator.py`
- `configs/afab/AF2_yolo26n_chaotic_amplitude.yaml`
- `docs/FARUQ_V3_AF2_DIRECT_FROM_PRETRAINED_PROTOCOL_2026-08-24.md`
- `docs/FARUQ_V3_AF2_SPECTRAL_FACTORIZATION_PROTOCOL.md`

---

## 2. Xu et al. (2025): exact evidence currently recertified

### 2.1 Fourier preliminaries

**Locator:** p. 3, §3.1.1, Eq. (1).

Xu et al. define the 2D-DFT for an image and use it as the frequency-domain foundation for the subsequent LFDet design. This supports using a standard 2D Fourier representation in the explanatory derivation.

**Boundary:** the paper's Eq. (1) is a mathematical definition. The repository's use of `torch.fft.fft2(..., norm="ortho")` and `fftshift` is an implementation choice and must not be attributed verbatim to Eq. (1).

### 2.2 Position of AFAB in LFDet

**Locator:** p. 4, §3.3, Fig. 1 and Fig. 2.

The paper places AFAB in the **data/input space**, while CGFI acts in feature space and FTIF targets the fine-grained classification pathway. Fig. 2 shows AFAB as:

`patch-wise DFT -> patch-specific adaptive high-pass filter -> patch-specific chaotic amplitude suppressor -> patch-wise iDFT`.

**Critical distinction:** the full AFAB described by Xu et al. contains two frequency subcomponents:

- AFAB-1: patch-specific adaptive high-pass filter;
- AFAB-2: patch-specific chaotic amplitude suppressor.

The thesis `mode=af2` frontend transfers the **AFAB-2-like angular amplitude mechanism**, not the complete AFAB-1+AFAB-2 sequence.

Therefore:

- do not call the thesis operator "the full AFAB";
- do not attribute AFAB-1 radial high-pass behavior to pure AF2;
- `radius_ratio` is inactive for pure `mode=af2`.

### 2.3 Patch-wise DFT and patch size

**Locator:** p. 5, §3.3.1.

Directly supported by the parent paper:

- DFT is restricted to local image patches rather than applied only globally;
- a sliding window is used;
- patch size is fixed to `m=32` in the parent method;
- a large patch overlap is used to alleviate discontinuity/pseudo high-frequency effects at patch boundaries;
- patch-wise iDFT reconstructs the spatial-domain response after frequency processing.

**Boundary:** the paper supports the *concept* of large overlap, but the thesis value `overlap=0.50`, exact stride `16`, `replicate` padding, and normalized fold averaging are frozen repository decisions unless a specific paper line states otherwise.

### 2.4 Angular density

**Locator:** §3.3.3, Eq. (9). The exact printed page number must be re-captured in the final page-perfect citation pass; do not invent it.

Parent equation:

\[
D_i^P(\theta)
=\sum_r A_i^P(r\cos\theta,r\sin\theta),
\qquad \theta\in[0,360^\circ).
\]

The parent paper explains angular density as a direction-wise aggregation of amplitude/frequency intensity and suppresses directions with low normalized density.

**Boundary:** the continuous angular domain is parent-paper evidence. The thesis implementation freezes it into `360` bins using floor-to-bin discretization; that exact discretization is a repository transfer choice.

### 2.5 Entropy-conditioned threshold and angular suppression

**Parent-method evidence:** §3.3.3 states that the information entropy of the angular density is used to generate a patch-specific adaptive threshold, after which low-density directions are suppressed and the adjusted amplitude is combined with the original phase before iDFT.

Repository implementation explicitly maps:

- `af2_entropy_threshold(...)` to LFDet AFAB-2 Eq. (10)–(11);
- `_af2_weight(...)` to LFDet Eq. (9)–(13).

Frozen implementation:

\[
H=-\sum_k p(k)\log(p(k)+\varepsilon),
\]

\[
\tau=\frac{\gamma}{1+\exp(-H)},
\qquad \gamma=0.10,
\]

followed by max-normalized angular density and hard suppression below threshold.

**Open locator gate:** during this audit, the primary PDF retrieval directly exposed Eq. (9) and the prose describing entropy/threshold/phase preservation, but did not expose a page-perfect image/text capture for every printed Eq. (10)–(13). Until that page is directly re-captured, proposal prose may state the equations as the **frozen AF2 implementation mapped by repository annotations to LFDet Eq. (9)–(13)**, but must not fabricate an exact page number for Eq. (10)–(13).

### 2.6 Phase preservation and reconstruction

**Locator:** §3.3.3.

Direct parent-method support:

- amplitude is the component deliberately adjusted by AFAB-2;
- original phase is retained;
- adjusted amplitude plus original phase are reconstructed with iDFT.

**Boundary:** repository `ifft2(...).real`, `fold` reconstruction, per-image/per-channel min-max normalization, and the exact residual formula are implementation-level transfer choices unless explicitly stated in the paper.

### 2.7 Component ablation and non-additivity

**Locator:** p. 13, §4.4.1, Table 6.

For MAR20 mAP50:

- baseline: 82.90;
- AFAB-1: 84.04;
- AFAB-2: 84.21;
- AFAB-1 + AFAB-2: 83.56.

For FAIRPlane11-2.0 mAP50:

- baseline: 45.20;
- AFAB-1: 45.40;
- AFAB-2: 45.64;
- AFAB-1 + AFAB-2: 45.30.

This is legitimate parent-paper evidence that the two AFAB subcomponents are **not automatically additive**.

**Allowed use in thesis:** justify factorized/controlled optimization rather than automatic stacking.

**Forbidden use:** do not claim these aircraft results prove the same interaction on coffee.

### 2.8 Overlap sensitivity

**Locator:** p. 17, Fig. 9 and Table 12.

The paper compares overlap ratios `0.5` and `0.75` and shows dataset-dependent accuracy/speed trade-offs.

**Allowed use:** support treating overlap as a meaningful sensitivity/design variable rather than a universal constant.

**Forbidden use:** do not claim `0.50` is globally optimal.

---

## 3. Repository-specific AF2 decisions

The following are **not to be silently attributed to Xu et al.** unless separately verified from the parent paper:

| Decision | Frozen repository behavior | Source |
|---|---|---|
| RGB handling | channels processed independently | `AFABInputEnhancer` docstring / implementation |
| Angular discretization | `[0,360°)` mapped to `angular_bins`; floor-to-bin | `_build_frequency_geometry` |
| DC mapping | DC maps to bin zero | `_build_frequency_geometry` comment |
| FFT numerical mode | FFT block forced to float32 under CUDA/AMP | `_filter_patch_chunk` |
| FFT normalization | `norm="ortho"` | `_filter_patch_chunk` |
| Padding | right/bottom `replicate` padding | `_pad_for_windows` |
| Overlap reconstruction | `fold` sum divided by `fold(ones)` divisor | `_recover_one` |
| Gate normalization | per-image, per-channel min-max | `minmax_spatial` |
| Residual image gate | `raw + raw * minmax(recovered)` | `afab_gate` |
| Chunk size | 128 | AF2 config |
| Exact overlap | 0.50 | AF2 config |
| Exact angular bins | 360 | AF2 config |
| Epsilon | `1e-8` | AF2 config |

The operator contains **zero learned parameters**, but this is not equivalent to zero compute cost.

---

## 4. Method-origin matrix for §3.5

| Method element in thesis | Parent paper | Repository adaptation | Study protocol |
|---|---:|---:|---:|
| Patch-wise local Fourier processing | YES | implemented | fixed |
| Parent patch size 32 | YES | same value | fixed |
| "large overlap" concept | YES | exact 0.50 | fixed |
| Angular density over `[0,360°)` | YES | 360 discrete bins | fixed |
| Entropy-adaptive threshold concept | YES | exact coded formula / `gamma=0.10` | fixed |
| Suppress low normalized angular-density directions | YES | hard threshold implementation | fixed |
| Preserve original phase during AFAB-2 reconstruction | YES | complex FFT implementation | fixed |
| Independent RGB processing | NOT specified in retrieved parent text | YES | fixed |
| Floor-to-bin angular discretization | NOT specified | YES | fixed |
| Replicate padding | NOT specified | YES | fixed |
| Fold/overlap averaging | paper says overlap/recovery, exact reducer not retrieved | YES | fixed |
| FP32 FFT under AMP | engineering choice | YES | fixed |
| Residual `I + I*MinMax(R)` | parent describes recovered-space gating; exact code form frozen here | YES | fixed |
| `radius_ratio=0.05` in pure AF2 | NO — belongs to AFAB-1 path | inactive | must not appear as AF2 factor |

---

## 5. YOLO26 primary-source hardening

### 5.1 Publication status

**Locator:** p. 1.

`[DET-01]` is arXiv:2606.03748v1, dated 2 June 2026. It must be described as a **preprint**, not as a Q1/Q2 journal paper.

### 5.2 Detector methodology

**Locator:** p. 5, §3, §3.1, §3.2, §3.2.1; Fig. 2; Table 1.

Directly supported by the primary paper:

- YOLO26 builds on YOLO11;
- native dual-head design with one-to-many and one-to-one paths;
- NMS-free one-to-one inference path;
- DFL-free direct box regression;
- training methodology introduces MuSGD, Progressive Loss, and STAL.

### 5.3 Full architecture schematic

**Locator:** supplementary p. 22, Fig. S1; supplementary p. 23, Fig. S2.

The supplement shows the shared backbone/neck, P3/P4/P5 scales, detection heads, and constituent blocks.

**Boundary:** the thesis uses **YOLO26n P3–P5 as a fixed detector family**. It is not proposing a new YOLO26 backbone/neck/head.

### 5.4 Paper recipe versus thesis recipe

Do not conflate the YOLO26 paper's benchmark training methodology with the thesis's actual experiment configuration.

The thesis frozen schedule comes from repository protocol/config:

```text
epochs        = 50 max
imgsz         = 640
batch         = 16
workers       = 2
patience      = 15
optimizer     = auto
pretrained    = true
cache         = false
close_mosaic  = 10
max_det       = 500
deterministic = true
```

Therefore, the proposal should say:

- **architecture/source model:** according to `[DET-01]` and official checkpoint;
- **coffee experiment schedule:** according to the frozen repository protocol.

Do not write that the thesis trains with MuSGD merely because MuSGD is a YOLO26 paper contribution if the actual frozen run uses `optimizer=auto`.

---

## 6. Direct-confirmation fairness provenance

Source: `docs/FARUQ_V3_AF2_DIRECT_FROM_PRETRAINED_PROTOCOL_2026-08-24.md`.

Frozen fairness facts:

- both arms start from the exact same official `yolo26n.pt` artifact;
- source checkpoint SHA-256 is frozen;
- both arms build the 21-class target detector with matched RNG initialization;
- persistent detector state is checked for exact equality before training;
- detector parameter counts are equal;
- AF2 contributes zero learned parameters;
- the only intended treatment difference is the deterministic input operator;
- train schedule is matched;
- validation is used for development diagnostics;
- locked test remains closed during method selection.

These claims are study-protocol facts, not claims from the YOLO26 or LFDet papers.

---

## 7. Optimization genealogy provenance

Source: `docs/FARUQ_V3_AF2_SPECTRAL_FACTORIZATION_PROTOCOL.md`.

Historical factorization changes one factor at a time from `AF2C`:

- `AF2WIN` — windowing;
- `AF2ORI` — orientation factorization;
- `AF2POL` — radial × orientation factorization;
- `AF2SOFT` — threshold factorization;
- `AF2LUM` — channel/luminance factorization.

`PCG1` and `WAV1` are mechanistic alternatives and must not be presented as merely additional AF2 toggles.

**Genealogy boundary:** historical factorization used seed-matched coffee-trained D0 parents. Final/direct confirmation uses the official pretrained YOLO26n source. These evidence families must remain visually and verbally separate.

---

## 8. Claim guardrails for the final proposal

Use:

- "AF2 is adapted from the AFAB-2 angular amplitude-suppression mechanism of Xu et al.";
- "the implementation freezes additional transfer decisions such as RGB-wise processing, angular discretization, overlap reconstruction, and residual gating";
- "factorized optimization evaluates one structural decision at a time";
- "YOLO26n is held fixed internally so the treatment is input preprocessing";
- "parameter-free does not mean compute-free".

Avoid:

- "AF2 is identical to the complete AFAB";
- "AF2 is a generic high-pass filter";
- "radius_ratio is an AF2 parameter";
- "Xu et al. proved the method works for coffee";
- "YOLO26 paper training recipe is identical to the thesis schedule";
- "Eq. (10)–(13) are on page X" until the exact primary PDF page is directly re-captured;
- "localization is solved" based only on proposal-accessibility diagnostics.

---

## 9. Remaining recertification gate

The only material page-level AF2 locator still open after this pass is a direct page-perfect capture of the printed Xu et al. **Eq. (10)–(13)** block. The mathematical implementation is already traceable to the repository annotations and parent §3.3.3 description, but the proposal archive should not invent a page number.

This is a citation-location gap, not an unresolved algorithm-definition gap.