# Faruq-v3 WAV1 Mechanism Factorization Protocol

Date frozen: 2026-08-19  
Status: **frozen before training**

Repository branch: `agent/wav1-mechanism-factorization`

Dataset: Faruq-v3 grouped development split only.  
Primary screening seed: **seed 42**.  
Native detector: YOLO26n-P3.  
Initialization: seed-matched D0 seed-42 checkpoint.  
Faruq **locked test remains closed**.

## Scientific question

The completed WAV1 paired-confirmation study established that the frozen standalone two-level Haar frontend is validation-robust versus D0FT across seeds 42/123/2026. This new study does not retune, redefine, or reopen that result. It asks a narrower mechanistic question:

> Which component of the parameter-free WAV1 cue explains the observed validation gain, especially the lower-tail gain?

The study separates three hypotheses:

1. **Wavelet specificity** — multiscale Haar organization contributes beyond a generic local-detail/high-pass cue.
2. **Scale contribution** — level-1 and level-2 detail provide complementary information rather than one level explaining nearly all of the effect.
3. **Scale equalization** — WAV1's per-level min-max normalization before fusion materially contributes to the effect.

This is an explanatory study, not a model-championship search. An arm can be scientifically informative without beating WAV1.

## Immutable positive reference

`WAV1_REF` delegates directly to the already-confirmed `af2_spectral/WAV1` implementation. It is not given a training config and must not be retrained in this study.

The frozen reference is:

\[
Y=0.2126R+0.7152G+0.0722B,
\]

followed by a two-level orthonormal Haar DWT. At level \(l\), the three detail bands are collapsed into

\[
D_l=\sqrt{LH_l^2+HL_l^2+HH_l^2+\epsilon}.
\]

Each level is resized to input resolution and min-max normalized independently. The two normalized levels are averaged, the recovered cue is min-max normalized again by the shared AFAB gate, expanded across RGB, and applied as

\[
x'=x+x\odot N(c(x)).
\]

No learned frontend parameters are introduced.

## Frozen causal arms

All trainable-screen arms keep the same luminance conversion, the same RGB residual gate, the same YOLO26n-P3 detector, the same D0 initialization, and the same training schedule. Only the cue generator/fusion factor named below changes.

| Arm | Frozen change | Mechanistic question |
|---|---|---|
| `WAV1_REF` | Exact confirmed WAV1 implementation; reference only | Positive reference |
| `HP1` | Fixed 3x3 binomial-smoothed luminance residual, absolute high-pass magnitude | Is generic local-detail enhancement sufficient? |
| `WAV_L1` | Only level-1 Haar detail energy | Does fine-scale detail explain the effect? |
| `WAV_L2` | Only level-2 Haar detail energy from `LL1` | Does coarser detail explain the effect? |
| `WAV_RAWFUSE` | Fuse raw resized level-1 + level-2 detail, then normalize once | Is per-level equalization important? |

### HP1 control

`HP1` deliberately contains no validation-tuned sigma. It uses the fixed separable binomial low-pass kernel

\[
\frac{1}{16}
\begin{bmatrix}
1&2&1\\
2&4&2\\
1&2&1
\end{bmatrix}
\]

and defines the cue before the common gate as

\[
D_{HP}=|Y-K*Y|.
\]

This is a generic local high-pass control. It is not claimed to reproduce any specific published LoG implementation.

### WAV_L1 and WAV_L2

The frozen level cues are

\[
C_{L1}=N(\tilde D_1),
\qquad
C_{L2}=N(\tilde D_2),
\]

where \(\tilde D_l\) is the bilinearly resized raw detail-energy map at level \(l\).

### WAV_RAWFUSE

The confirmed WAV1 equalizes each scale before fusion:

\[
C_{WAV1}=N\left(\frac{N(\tilde D_1)+N(\tilde D_2)}{2}\right).
\]

`WAV_RAWFUSE` removes only the per-scale equalization:

\[
C_{RAW}=N(\tilde D_1+\tilde D_2).
\]

Because the common AFAB gate performs the final min-max normalization, multiplying the raw sum by a positive constant would not change the final cue.

## Frozen training schedule

Each new arm uses the same schedule as the confirmed WAV1 run:

- epochs: 50
- imgsz: 640
- batch: 16
- workers: 2
- patience: 15
- optimizer: auto
- pretrained: false
- cache: false
- close_mosaic: 10
- max_det: 500
- save_period: 1
- deterministic seed-42 training

No arm may tune the high-pass kernel, wavelet level, interpolation rule, normalization rule, color conversion, gate, model architecture, or training hyperparameters after validation results are observed.

## Static gate before training

Before any arm trains, the static audit must PASS. It verifies:

- `WAV1_REF` is bitwise equal on CPU to the already-confirmed WAV1 operator;
- all frontends are finite, active, shape/dtype preserving, parameter/state free, CPU-bitwise repeatable, and provide finite gradients to the input;
- the D0 checkpoint hash is recorded;
- test access remains unauthorized.

If any condition fails, training is blocked.

## Stage-1 screening

Only `HP1`, `WAV_L1`, `WAV_L2`, and `WAV_RAWFUSE` train at seed 42. D0FT seed42 and the completed WAV1 seed42 result are reused as frozen references.

The headline metrics are:

- Macro mAP50-95;
- Bottom-3 class mAP50-95;
- Worst-class mAP50-95;
- per-class AP deltas when available;
- latency as a reported engineering metric, not a scientific selection criterion.

This study does **not** require a causal arm to beat WAV1. Explanatory retention is based on how much of the already-observed WAV1-vs-D0FT gain is preserved and whether the per-class direction is consistent with the proposed explanation.

For headline metric \(T\), report

\[
R_T(M)=\frac{T(M)-T(D0FT)}{T(WAV1)-T(D0FT)}.
\]

This ratio is descriptive. It is not used when the WAV1 denominator is zero or changes sign.

## Pre-frozen interpretation tree

- If `HP1` preserves most of the WAV1 lower-tail gain and has a similar per-class delta pattern, the wavelet-specific claim is weakened; the supported mechanism becomes generic local-detail modulation.
- If `WAV_L1` explains nearly all of the WAV1 effect while `WAV_L2` is weak, two-level/multiscale necessity is not supported.
- If `WAV_L2` explains nearly all of the effect while `WAV_L1` is weak, the interpretation shifts toward coarser structural detail.
- If WAV1 remains materially stronger than both single levels, multiscale complementarity is supported descriptively.
- If `WAV_RAWFUSE` is close to WAV1, per-level equalization is not a necessary explanation.
- If WAV1 is materially stronger than `WAV_RAWFUSE`, scale equalization remains a supported component of the mechanism.

No orientation-specific `LH`/`HL`/`HH` study is opened from this protocol unless a single wavelet level is first supported as the dominant mechanism.

## Follow-up rule

After seed-42 screening, at most one mechanistic survivor may be opened for a separate seed-123/2026 paired confirmation protocol. That later protocol must be frozen before additional training. This document does not authorize those runs automatically.

A later input-vs-classification-placement study is also a separate protocol and is not authorized by the current factorization screen.

## Test lock and reporting boundary

- Faruq locked test is not restored, read, or reopened.
- No post-result tuning is allowed inside this protocol.
- Existing WAV1 confirmation evidence is not rewritten regardless of the outcome here.
- A failed mechanistic explanation does not invalidate the already-confirmed empirical WAV1-vs-D0FT result; it only limits how that result may be interpreted.
