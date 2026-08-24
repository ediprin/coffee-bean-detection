# Faruq-v3 AF2 → FFAB2 experiment chain result ledger

Date: 2026-08-24

Status: **completed through parent-preserving follow-up. Test remained locked in every experiment recorded here.**

This ledger consolidates the AF2+FFAB2 experiments that were discussed and completed across four distinct training/evaluation regimes. It intentionally keeps their comparators separate.

## Executive result

| Study | Frozen comparator | Macro Δ | Bottom-3 Δ | Worst Δ | Decision |
|---|---|---:|---:|---:|---|
| Staged continuation AF2FFAB2 | AF2FFA0 continuation control | +1.44 pp | +2.83 pp | +2.27 pp | **PASS** |
| Matched from-start AF2FFAB2FS | AF2FS | -0.0687 pp | +1.4151 pp | +2.1082 pp | **REJECT** (Macro gate) |
| Selective runtime diagnosis (best Macro: `beta_0.25`) | AF2FS | -0.0605 pp | +1.3914 pp | +2.1180 pp | **NO_RUNTIME_CANDIDATE_PASSES_GATE** |
| AF2 parent-preserving FFAB2 | frozen completed AF2FS parent | +0.0057 pp | -0.0223 pp | -0.0216 pp | **REJECT** |

The continuation PASS does **not** mean FFAB2 universally improves the original AF2 checkpoint. Its causal comparator was the equally continued zero-information control. The matched from-start experiment is the direct test of AF2+FFAB2 versus AF2 under the same from-start schedule, and it failed the frozen Macro-improvement gate. The parent-preserving follow-up then tested whether FFAB2 alone can add value without allowing the AF2 parent to move; it also failed.

## 1. Staged continuation: AF2FFAB2 vs AF2FFA0

Both arms started from the same seed-matched completed AF2 checkpoint and continued for 30 epochs. AF2FFA0 was the capacity/optimization-matched zero-information control.

| Seed | AF2FFA0 Macro / B3 / Worst | AF2FFAB2 Macro / B3 / Worst |
|---:|---:|---:|
| 42 | 88.89% / 80.84% / 77.53% | **88.89% / 82.11% / 80.49%** |
| 123 | 86.89% / 74.73% / 70.57% | **88.41% / 78.03% / 71.46%** |
| 2026 | 85.52% / 77.48% / 75.99% | **88.33% / 81.41% / 78.95%** |

Three-seed means: AF2FFA0 = **87.10 / 77.68 / 74.69**; AF2FFAB2 = **88.54 / 80.52 / 76.96**. Mean deltas = **+1.44 / +2.83 / +2.27 pp**, all three metrics positive in 3/3 seeds. Frozen decision: **PASS**.

Descriptive only versus pre-continuation original AF2 (87.94 / 79.37 / 78.15): AF2FFAB2 was about **+0.60 / +1.15 / -1.19 pp**. This cross-regime contrast was not the frozen causal comparison.

## 2. Matched from-start: AF2FFAB2FS vs AF2FS

Both arms used the same seed-matched D0 start and the same 50-epoch from-start schedule. This experiment tested whether FFAB2 is a global AF2 upgrade when trained jointly from the start.

| Seed | AF2FS Macro / B3 / Worst | AF2FFAB2FS Macro / B3 / Worst | Δ Macro / B3 / Worst |
|---:|---:|---:|---:|
| 42 | 88.1973% / 80.0428% / 79.3470% | 87.5340% / 81.3139% / 80.1778% | -0.6633 pp / +1.2712 pp / +0.8308 pp |
| 123 | 87.4176% / 75.9345% / 72.8440% | 87.7373% / 76.0288% / 70.8313% | +0.3197 pp / +0.0942 pp / -2.0127 pp |
| 2026 | 87.2436% / 78.9019% / 73.2075% | 87.3812% / 81.7818% / 80.7141% | +0.1375 pp / +2.8799 pp / +7.5066 pp |

Three-seed means: AF2FS = **87.6195% / 78.2931% / 75.1328%**; AF2FFAB2FS = **87.5508% / 79.7082% / 77.2411%**.

Mean deltas = **-0.0687 pp / +1.4151 pp / +2.1082 pp**. Macro improved in 2/3 seeds, Bottom-3 in 3/3, Worst in 2/3.

Frozen criteria required Macro mean gain ≥ +0.50 pp plus seed consistency, Bottom-3 mean gain ≥ +0.50 pp plus seed consistency, and non-negative/consistent Worst. Every tail criterion passed, but the Macro mean-gain criterion failed. Final decision: **REJECT**; `STOP_FFAB2_UPGRADE_CLAIM`. DCT was not run.

## 3. Selective-refinement diagnosis

This stage reused the completed AF2FFAB2FS checkpoints and performed validation-only inference ablations. It did not train a new candidate and did not open test.

Frozen diagnostic authorization gate: Macro Δ ≥ 0.25 pp and ≥2/3 positive seeds; Bottom-3 Δ ≥ 0.50 pp and ≥2/3; Worst Δ ≥ 0.00 pp and ≥2/3.

### All 19 runtime variants

| Variant | Family | Δ Macro | Δ Bottom-3 | Δ Worst | Positive seeds M/B3/W | Eligible |
|---|---|---:|---:|---:|---:|---|
| `beta_0.00` | strength | -0.0668 pp | +1.3807 pp | +2.0861 pp | 2/3/2 | NO |
| `beta_0.25` | strength | -0.0605 pp | +1.3914 pp | +2.1180 pp | 2/3/2 | NO |
| `beta_0.50` | strength | -0.0631 pp | +1.4051 pp | +2.0878 pp | 2/3/2 | NO |
| `beta_0.75` | strength | -0.0665 pp | +1.4134 pp | +2.1082 pp | 2/3/2 | NO |
| `beta_1.00` | strength | -0.0687 pp | +1.4151 pp | +2.1082 pp | 2/3/2 | NO |
| `levels_P3` | levels | -0.0698 pp | +1.3896 pp | +2.1082 pp | 2/3/2 | NO |
| `levels_P4` | levels | -0.0658 pp | +1.4062 pp | +2.0861 pp | 2/3/2 | NO |
| `levels_P5` | levels | -0.0695 pp | +1.3807 pp | +2.0861 pp | 2/3/2 | NO |
| `levels_P3_P4` | levels | -0.0687 pp | +1.4151 pp | +2.1082 pp | 2/3/2 | NO |
| `levels_P3_P5` | levels | -0.0698 pp | +1.3896 pp | +2.1082 pp | 2/3/2 | NO |
| `levels_P4_P5` | levels | -0.0658 pp | +1.4062 pp | +2.0861 pp | 2/3/2 | NO |
| `parent_mix_0.25` | parent_residual | -0.0605 pp | +1.3914 pp | +2.1180 pp | 2/3/2 | NO |
| `parent_mix_0.50` | parent_residual | -0.0631 pp | +1.4051 pp | +2.0878 pp | 2/3/2 | NO |
| `parent_mix_0.75` | parent_residual | -0.0665 pp | +1.4134 pp | +2.1082 pp | 2/3/2 | NO |
| `parent_mix_1.00` | parent_residual | -0.0687 pp | +1.4151 pp | +2.1082 pp | 2/3/2 | NO |
| `ambiguity_margin_0.05` | ambiguity | -0.0675 pp | +1.3785 pp | +2.0750 pp | 2/3/2 | NO |
| `ambiguity_margin_0.10` | ambiguity | -0.0693 pp | +1.3684 pp | +2.0447 pp | 2/3/2 | NO |
| `ambiguity_margin_0.15` | ambiguity | -0.0699 pp | +1.3684 pp | +2.0447 pp | 2/3/2 | NO |
| `ambiguity_margin_0.20` | ambiguity | -0.0683 pp | +1.3752 pp | +2.0652 pp | 2/3/2 | NO |

Final diagnostic decision: **`NO_RUNTIME_CANDIDATE_PASSES_GATE`**; `training_authorized=false`; `test_opened=false`; next recorded state: **`PARENT_PRESERVING_ARCHITECTURE_REMAINS_IMPLEMENTED_BUT_NOT_AUTHORIZED`**.

The best runtime condition by Macro was `beta_0.25`, but its mean Macro delta was still negative, so no Stage-B selective retraining was authorized.

### Per-class FFAB2FS − AF2FS aggregate

| Class | Mean Δ AP | Improved seeds |
|---|---:|---:|
| `biji_berlubang_satu` | +2.6165 pp | 3/3 |
| `biji_muda` | +2.3603 pp | 1/3 |
| `kulit_tanduk_ukuran_sedang` | +2.1369 pp | 2/3 |
| `biji_pecah` | +1.5847 pp | 3/3 |
| `biji_berlubang_lebih_satu` | +1.1618 pp | 2/3 |
| `biji_coklat` | +1.1250 pp | 2/3 |
| `tanah_batu_ranting_sedang` | +1.0523 pp | 2/3 |
| `biji_bertutul_tutul` | +0.6284 pp | 2/3 |
| `tanah_batu_ranting_kecil` | +0.0315 pp | 1/3 |
| `kopi_gelondong` | -0.0221 pp | 2/3 |
| `biji_normal` | -0.0247 pp | 2/3 |
| `kulit_kopi_ukuran_sedang` | -0.3183 pp | 1/3 |
| `biji_hitam` | -0.4248 pp | 1/3 |
| `kulit_tanduk_ukuran_besar` | -0.7944 pp | 1/3 |
| `kulit_kopi_ukuran_besar` | -0.8408 pp | 2/3 |
| `kulit_tanduk_ukuran_kecil` | -0.9575 pp | 2/3 |
| `biji_hitam_pecah` | -1.0844 pp | 1/3 |
| `tanah_batu_ranting_besar` | -1.9667 pp | 1/3 |
| `biji_hitam_sebagian` | -2.3227 pp | 1/3 |
| `kulit_kopi_ukuran_kecil` | -2.5791 pp | 1/3 |
| `biji_berkulit_tanduk` | -2.8045 pp | 1/3 |

Only two classes improved in all 3/3 seeds: **`biji_berlubang_satu` (+2.6165 pp)** and **`biji_pecah` (+1.5847 pp)**. No class regressed in all 3/3 seeds (`consistent_harm=[]`). Per seed, the number of improved/regressed classes was 11/10 (seed 42), 11/10 (seed 123), and 12/9 (seed 2026).

## 4. What the selective diagnosis supports

- Turning the learned FFAB2 residual off at inference (`beta_0.00`) did **not** restore AF2FS Macro: Δ Macro remained -0.0668 pp while Bottom-3/Worst remained +1.3807/+2.0861 pp.
- Strength scaling from beta 0 to 1 changed Macro by only about 0.008 pp end-to-end; all strength variants retained the same broad tail-gain/Macro-flat pattern.
- P3/P4/P5 bypass combinations were similarly close; no level subset recovered positive mean Macro.
- Parent-logit interpolation and ambiguity-gated runtime correction also failed the Macro authorization gate.
- Therefore the earlier hypothesis that *inference-time FFAB2 strength alone* was the main cause of Macro loss is not supported.
- A plausible explanation is training co-adaptation: the AF2FFAB2FS backbone/neck/classifier trajectory changed during joint optimization, so setting the adapter residual to zero after training does not recreate the AF2FS checkpoint. This is a **mechanistic hypothesis consistent with the ablation**, not causal proof.

## 5. Parent-preserving FFAB2: AF2FFAPR1 vs frozen AF2FS

This follow-up started from the completed seed-matched AF2FS checkpoint, froze the AF2 frontend, backbone, neck, native regression/classification paths, and BatchNorm state, and trained only the FFAB2 adapters. The research question was whether frequency-conditioned residual adaptation can add value without allowing the mature AF2 parent to co-adapt.

Frozen decision gate: Macro mean gain >= +0.50 pp with >=2/3 improved seeds; Bottom-3 mean gain >= +0.50 pp with >=2/3 improved seeds; Worst mean delta >= 0 with >=2/3 improved seeds.

| Metric | Frozen AF2FS parent | AF2FFAPR1 | Mean Δ | Improved seeds |
|---|---:|---:|---:|---:|
| Macro mAP50-95 | 87.6195% | 87.6252% | **+0.0057 pp** | 2/3 |
| Bottom-3 class mAP50-95 | 78.2931% | 78.2708% | **-0.0223 pp** | 0/3 |
| Worst-class mAP50-95 | 75.1328% | 75.1112% | **-0.0216 pp** | 0/3 |

Per-seed deltas:

- seed 42: Macro +0.0044 pp, Bottom-3 0.0000 pp, Worst 0.0000 pp;
- seed 123: Macro -0.0003 pp, Bottom-3 0.0000 pp, Worst 0.0000 pp;
- seed 2026: Macro +0.0130 pp, Bottom-3 -0.0668 pp, Worst -0.0648 pp.

Frozen criteria:

- `macro_mean_gain_at_least_0_5pp=false`
- `macro_improves_at_least_2_of_3=true`
- `bottom3_mean_gain_at_least_0_5pp=false`
- `bottom3_improves_at_least_2_of_3=false`
- `worst_mean_not_lower=false`
- `worst_improves_at_least_2_of_3=false`

Final decision: **`REJECT`**.

Next: **`STOP_PARENT_PRESERVING_FFAB2_ROUTE`**.

The result shows that FFAB2 alone did not provide meaningful incremental headroom on a frozen mature AF2 representation. Macro was effectively unchanged and the lower-tail metrics were slightly lower. This is consistent with the broader observation that the FFAB2 tail gains seen during joint adaptation are not reproduced when only the FFAB2 residual is allowed to learn. This interpretation is consistent with the experiment chain but is not proof of a unique causal mechanism.

Important provenance note: the completed decision artifact recorded `zero_control: null`, so this final parent-preserving run is a direct AF2FFAPR1-versus-frozen-AF2FS test, not a completed spectral-versus-zero-information residual comparison. This does not change the REJECT decision because the spectral candidate itself failed the preregistered parent-improvement gate.

Detailed result: `docs/FARUQ_V3_AF2_FFAB2_PARENT_PRESERVING_RESULT_2026-08-24.md`.

## 6. Experiment-chain conclusion

The evidence across the four regimes is now:

1. **Continuation:** FFAB2 beats an equally continued zero-information control and improves Macro/Bottom-3/Worst under that matched continuation regime.
2. **From-start:** the lower-tail benefit persists, but the global Macro-upgrade criterion does not replicate.
3. **Selective runtime diagnosis:** changing adapter strength, level routing, parent mixing, or ambiguity gating at inference does not recover a positive Macro mean delta.
4. **Frozen parent:** FFAB2 alone is effectively neutral on Macro and slightly negative on Bottom-3/Worst, so it does not provide a parent-preserving upgrade.

The disciplined conclusion is therefore: **FFAB2 is not retained as a global or parent-preserving upgrade of AF2. Its strongest evidence remains a regime-dependent lower-tail benefit when optimized jointly/continuationally with the network.**

## Evidence / provenance

- Continuation result: `docs/FARUQ_V3_AF2_FFA_B2_PAIRED_CONFIRMATION_RESULT_2026-08-22.md`.
- From-start runner/decision: `run_faruq_v3_af2_ffa_from_start_decision.py`; result values reproduced by the completed `AF2FFAB2FS` checkpoint under `beta_1.00` in `selectivity_analysis.json`.
- Selective-refinement raw artifact: `selectivity_analysis.json`, format `coffee_detector.af2_ffa.selectivity_analysis.v1`.
- Parent-preserving final result: `docs/FARUQ_V3_AF2_FFAB2_PARENT_PRESERVING_RESULT_2026-08-24.md`.
- Parent-preserving decision artifact format: `coffee_detector.af2_ffa.parent_preserving_decision.v1`.
- Test remained locked for all experiments summarized in this ledger.
