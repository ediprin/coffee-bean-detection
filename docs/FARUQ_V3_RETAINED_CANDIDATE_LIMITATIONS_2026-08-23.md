# Faruq-v3 retained-candidate limitations and next analysis

Date: 2026-08-23
Status: evidence consolidation; no training and no new test access

## Purpose

This note prevents a `RETAIN` label or a high seed-42 score from being treated
as universal superiority. It consolidates the empirical limitations of every
retained direction and freezes the next step as a no-training multi-objective
and complementarity analysis.

Evidence levels are kept separate:

- **confirmed**: passed a frozen paired multi-seed gate;
- **discovery**: retained by a seed-42 screening gate only;
- **control**: useful optimization/capacity observation, not evidence for a
  proposed mechanism.

## Confirmed or advanced candidates

| Candidate | Evidence | Main strength | Empirical limitation |
|---|---|---|---|
| Original AF2 | Paired three-seed PASS versus D0FT; positive reused-test and Coffee Standard directions | Best overall evidence balance; Macro 87.94%, Bottom-3 79.37%, Worst 78.15% across validation seeds; strongest Coffee Standard direction | Parameter-free but not compute-free: about 1.75x median latency and 0.58x throughput versus D0FT. Controlled illumination failed overall: warm/cool improved, but exposure, contrast, and shadow often regressed. Synthetic-density absolute AP remained very low. Mechanism attribution is classification-dominant, not raw-localization improvement. |
| IGEM1 | Paired three-seed PASS versus D0FT | Strong lower-tail geometry-aware alternative; 87.71/79.27/77.74 three-seed means | Slightly below AF2 on all aggregate validation headline metrics; Coffee Standard Macro only 14.04%; no locked-test confirmation; geometry information can be domain- and scale-dependent. |
| ACMC1 | Paired three-seed validation PASS; locked test executed | Improved D0FT validation and lower-tail metrics | Locked-test conclusion was `NOT_CONFIRMED`; Adrian external Macro was lower than D0FT in 3/3 seeds. It is not a demonstrated cross-domain solution. |
| GEO1 | Seed-42 control screen retained; paired three-seed geometry mechanism reported as confirmed in its branch | Very small geometry adapter and positive size-family signal | Family effect is heterogeneous: `kulit_kopi` was negative in 3/3 seeds while `kulit_tanduk` was positive. Synthetic density was below D0FT on average and Coffee Standard gain was small; predicted geometry inherits camera-scale/domain bias. |
| AF2FFAB2 | Paired three-seed PASS versus matched AF2FFA0 continuation control | Repeatable Pareto gain versus its matched control: +1.44 Macro, +2.83 Bottom-3, +2.27 Worst points | It does not universally dominate original AF2. Descriptively it is about +0.60 Macro and +1.15 Bottom-3 but -1.19 Worst versus original AF2; it also adds feature-level FFT work. Test and external robustness are not established. |

## Discovery-only retained candidates

| Candidate | Seed-42 Macro / Bottom-3 / Worst | Empirical limitation |
|---|---:|---|
| STB1 | 88.67 / 83.64 / 80.81 | Strong seed-42 score, but its paired capacity-causal confirmation failed because mean Macro gain over CMC0 was only +0.07 point. Seed 123 had a large tail drop. Coffee Standard Macro was 14.45%. Treat as a high-performing reference, not a confirmed shifted-window contribution. |
| AF1 | 87.94 / 80.07 / 77.05 | Inferior to AF2 on the key tail evidence; synthetic density improved only 2/4 conditions and Coffee Standard Macro was 14.31%. No multi-seed confirmation. |
| SAF1 | 87.34 / 81.33 / 80.34 | Strongest average synthetic-density direction, but lower in-domain Macro and Coffee Standard Macro of 13.77%. No multi-seed confirmation. Direct AF2+SAF joint training produced only a Pareto shift, not a clean win. |
| CPE0 | 86.91 / 77.36 / 74.50 | Relatively strong Coffee Standard Macro (15.82%) but clearly behind the leading in-domain candidates; external Bottom-3 and Worst remained near zero. No multi-seed confirmation. |
| CPE7 | 86.56 / 76.25 / 72.70 | Weaker than CPE0 in-domain, negative synthetic-density direction, and near-zero external tail. No multi-seed confirmation. |
| HVIP1 | 86.90 / 81.21 / 78.36 | Tail is promising but Macro is behind the leaders; no paired multi-seed or external confirmation. |
| PW1 | 86.36 / 78.84 / 76.62 | Moderate improvement but Pareto-dominated by stronger retained candidates; no paired multi-seed or external confirmation. |
| SG1/LPS1 | 86.12 / 79.14 / 76.35 | Lower Macro and external discrimination despite relatively high external recall. No multi-seed confirmation. |
| SEMAUX/LPS1 | 86.12 / 79.14 / 76.35 | Status conflict: the candidate-local gate rejected it, while the later common breadth gate retained it. It must never be cited as unqualified `RETAIN`. |
| FT1 | 87.72 / 80.66 / 80.24 | Only FTIF arm competitive across all three metrics, but absent from the canonical breadth snapshot and still seed-42 discovery evidence. |
| FT2 | 87.39 / 79.19 / 75.22 | Retained locally, but its lower tail is materially below FT1. No multi-seed confirmation. |
| FT3 | 87.50 / 78.41 / 69.37 | Retained locally on its own gate, but Worst-class AP is too weak for a lower-tail robustness claim. No multi-seed confirmation. |

## High numerical controls that are not proposed methods

`FCT0`, `AF2R0`, `AF2FT30`/`AF2CT30`, and `AF2FFA0` are continuation,
capacity, or zero-information controls. Their high validation numbers establish
that additional optimization can materially move the result. They do not
establish a new architectural mechanism and must not be ranked as thesis
methods.

The following proposed additions failed against their matched controls or
paired confirmation and are not retained methods: AF2R1, AF2CAL3, AF2_ORIENT,
AF2_RADIAL, AF2RCC1, DIDA-AF2, AF2STB1, AF2IGEM1, and AF2SAF1.

## Direct AF2 plus strong-model result

The completed direct joint-from-D0 pairs did not establish additive benefit:

| Pair versus standalone parent | Macro delta | Bottom-3 delta | Worst delta | Decision |
|---|---:|---:|---:|---|
| AF2STB1 - STB1 | -1.45 pt | -4.78 pt | -4.99 pt | REJECT |
| AF2IGEM1 - IGEM1 | -0.55 pt | -1.76 pt | -1.93 pt | REJECT |
| AF2SAF1 - SAF1 | +0.75 pt | -0.48 pt | -0.67 pt | REJECT; descriptive Pareto shift only |

Static wiring checks established that AF2 was active in training and inference
and that each strong architecture was instantiated as intended. These runs test
joint optimization from D0; they do not test parent-preserving fusion from the
already trained AF2 and strong checkpoints.

## Evidence conclusion and subsequent authorization

No retained candidate currently dominates all objectives simultaneously:
in-domain Macro, Bottom-3, Worst, seed stability, cross-domain performance,
synthetic density, illumination robustness, latency, and memory.

At the time this consolidation was written, the next authorized activity was
analysis only:

1. build one evidence-level-aware multi-objective atlas;
2. compute Pareto fronts without mixing controls with proposed methods;
3. measure per-image and per-class prediction complementarity among D0FT,
   AF2, IGEM1, SAF1, and optionally GEO1;
4. calculate oracle routing upper bounds before designing another fusion;
5. authorize a router or parent-preserving residual only if the existing
   predictions demonstrate material complementary headroom.

This note itself does not authorize training or additional test access. After
reviewing the direct-pair limitation, the user separately authorized the
parent-preserving SAF/IGEM experiment frozen in
`FARUQ_V3_AF2_PARENT_RESIDUAL_PROTOCOL_2026-08-23.md`. That later protocol
supersedes only the no-training sentence for its four named seed-42 arms; the
test lock and every other limitation above remain in force.
