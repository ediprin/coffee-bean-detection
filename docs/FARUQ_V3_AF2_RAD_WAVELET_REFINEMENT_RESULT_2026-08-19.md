# Faruq-v3 AF2 Radial-Wavelet Refinement Result

Date: 2026-08-19

Status: **COMPLETED — FAIL; KEEP AF2C AND STOP.**

This document records the completed seed-42 follow-up that was opened only after the AF2 spectral-factorization study suggested two mechanistic leads: the lower-tail gain of `AF2POL` and the strong standalone result of `WAV1`. The follow-up tested whether those leads could improve legacy AF2 directly without reopening the locked Faruq test.

## Frozen question

The study tested three parameter-free input-frontends against the historical seed-42 AF2 control (`AF2C`):

- `AF2RAD`: legacy AF2 directional processing retained at 360 angular bins, with only a 3-band radial decomposition added.
- `AF2WAV`: legacy AF2 cue fused in parallel with the exact two-level Haar wavelet cue used by `WAV1`, using parameter-free pointwise max fusion.
- `AF2RADWAV`: `AF2RAD` cue fused with the same wavelet cue using the same max operator.

The follow-up deliberately did **not** reuse AF2POL's Hann window or 16-bin orientation reduction. Therefore `AF2RAD` isolates radial structure inside legacy AF2 rather than reproducing the cumulative `WIN + ORI + POL` arm.

All arms used the same Faruq-v3 grouped development data, the same seed-42 D0 initialization, 50 epochs, validation-only evaluation, and no test access.

## Static audit

The Kaggle static audit passed before training. It established:

- D0 is the expected 21-class detector.
- `AF2C` is bitwise equal to legacy AF2.
- `AF2RAD` uses 3 radial bands and the legacy 360-bin AF2 angle map unchanged.
- The wavelet cue is bitwise equal to the already-tested `WAV1` cue.
- Max fusion never attenuates the AF2 cue pointwise.
- All three candidates are repeatable within the frozen tolerance, finite in forward/backward, parameter-free, and have no persistent state.
- Test access remained disabled.

Static audit decision: **PASS**.

## Seed-42 validation results

| Arm | Macro mAP50-95 | Bottom-3 | Worst | Median latency (ms) | Decision |
|---|---:|---:|---:|---:|---|
| **AF2C** | **88.197%** | 80.043% | 79.347% | historical control | KEEP |
| AF2RAD | 86.776% | 76.954% | 72.090% | 24.524 | REJECT |
| AF2WAV | 87.725% | **81.707%** | **80.817%** | 25.445 | REJECT |
| AF2RADWAV | 87.101% | 79.347% | 77.133% | 27.265 | REJECT |

### Delta versus AF2C

| Arm | Δ Macro | Δ Bottom-3 | Δ Worst |
|---|---:|---:|---:|
| AF2RAD | **-1.421 pp** | **-3.089 pp** | **-7.257 pp** |
| AF2WAV | **-0.472 pp** | **+1.664 pp** | **+1.470 pp** |
| AF2RADWAV | **-1.096 pp** | **-0.695 pp** | **-2.214 pp** |

The frozen retention gate required all of the following relative to AF2C:

1. Macro gain at least +0.50 percentage point.
2. Bottom-3 not lower.
3. Worst-class drop no greater than 1.00 point.

No arm passed the gate. `AF2RAD` and `AF2RADWAV` failed all or multiple headline criteria. `AF2WAV` improved both lower-tail metrics but lost 0.472 point Macro and therefore failed the primary Macro criterion.

Final decision object: **FAIL → `KEEP_AF2C_AND_STOP`**.

No seed-123/2026 confirmation, no retuning, and no locked-test evaluation are authorized for these three follow-up arms.

## Mechanistic interpretation

### 1. Radial structure alone does not explain AF2POL

The prior cumulative `AF2POL` arm had shown a small Macro loss but large Bottom-3/Worst gains. This follow-up directly tested the hypothesis that radial decomposition itself caused that behavior.

`AF2RAD`, which inserts radial structure into legacy 360-bin AF2 without the preceding Hann-window and 16-bin-orientation changes, performed substantially worse than AF2C on all three headline metrics. Therefore the earlier AF2POL lower-tail gain **cannot be attributed to radial decomposition alone**.

The defensible interpretation is that AF2POL's behavior depended on the cumulative interaction of `WIN + ORI + POL` (or another interaction created by that cumulative configuration), not a standalone radial benefit.

### 2. Standalone WAV1 does not transfer through AF2 fusion

The prior standalone `WAV1` result was 88.411% Macro, 83.276% Bottom-3, and 82.035% Worst, with about 15.92 ms median latency. In contrast, `AF2WAV` reached only 87.725% / 81.707% / 80.817% at about 25.45 ms.

Thus the useful standalone wavelet representation is **not preserved as a superior detector when fused with AF2 using the tested max-cue formulation**. Pointwise non-attenuation of the AF2 cue is an implementation property, not a guarantee that the downstream fine-tuned detector AP will be non-decreasing.

### 3. Radial + wavelet does not rescue either mechanism

`AF2RADWAV` was lower than AF2C on Macro, Bottom-3, and Worst, so the two ideas do not form a beneficial composition under the frozen formulation.

## Final research status

- Keep the previously confirmed AF2 evidence unchanged.
- Reject `AF2RAD`, `AF2WAV`, and `AF2RADWAV` as replacements for AF2C under this protocol.
- Do not claim that radial decomposition is independently responsible for AF2POL's lower-tail gains.
- Do not claim that WAV1 can be improved by direct cue fusion with AF2.
- Treat `AF2POL` and standalone `WAV1` as mechanistic/lower-tail observations only; neither authorizes post-hoc fusion or extra seeds under the completed protocols.
- Faruq locked test remains closed.

## Provenance

Training notebook commit: `35ec68dfc1af5fa27ee29ee2acc86b6b4aa914c2` on branch `agent/af2-rad-wavelet-refinement`.

Frozen protocol: `docs/FARUQ_V3_AF2_RAD_WAVELET_REFINEMENT_PROTOCOL.md`.

Machine-readable evidence: `docs/evidence/FARUQ_V3_AF2_RAD_WAVELET_REFINEMENT_RESULT_2026-08-19.json`.
