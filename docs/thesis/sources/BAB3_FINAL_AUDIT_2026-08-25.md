# Bab III Final Proposal Audit — 2026-08-25

Status: **final chapter-level audit before formal campus-document assembly**.

Scope: methodology logic, source provenance, experiment genealogy, optimization claims, figures, evaluation, and pilot/final evidence boundaries.

This audit does not launch or require any new experiment.

---

## 1. Audit verdict

\[
\boxed{\text{BAB III PROPOSAL DESIGN: PASS FOR FORMAL ASSEMBLY}}
\]

with one non-scientific remaining task:

- equation/figure typesetting proofread during DOCX/PDF assembly.

The previously open Xu AFAB-2 page locator is now closed by `BAB3_AF2_PAGE_LOCATOR_RECERTIFICATION_2026-08-25.md`.

---

## 2. Method identity and provenance

| Check | Verdict | Evidence / rule |
|---|---|---|
| AF2 positioned before detector | PASS | input-space frontend in proposal architecture |
| AF2 source family identified | PASS | adapted from Xu et al. AFAB-2 angular amplitude suppression |
| thesis AF2 separated from full AFAB | **PASS** | pure `mode=af2` excludes AFAB-1 radial high-pass |
| `radius_ratio` treated correctly | **PASS** | inactive in pure AF2; not an AF2 optimization factor |
| parent-paper vs repository decisions separated | PASS | method-origin matrix + source-hardening audit |
| AFAB-2 equation family page-located | **PASS** | Xu et al. pp. 5–6, §3.3.3, Eq. (9)–(13) |
| repository-specific formulas traceable | PASS | `operator.py` + AF2 config |
| parameter-free phrasing bounded | PASS | zero learned frontend params; compute cost still measured |

### Required final wording

Use:

> AF2 diadaptasi dari mekanisme angular amplitude suppression pada AFAB-2 Xu et al., kemudian ditransfer ke pipeline YOLO26 melalui sejumlah keputusan implementasi yang dibekukan pada repository.

Do not use:

> AF2 adalah AFAB lengkap Xu et al.

or:

> AF2 adalah high-pass filter.

---

## 3. YOLO26 baseline provenance

| Check | Verdict | Evidence / rule |
|---|---|---|
| primary baseline source identified | PASS | Jocher et al. 2026, arXiv:2606.03748v1 |
| publication status honest | PASS | **preprint**, not Q1/Q2 journal |
| detector architecture sourced from paper | PASS | §3 / supplementary Fig. S1–S2 |
| thesis does not claim new YOLO backbone/neck/head | PASS | detector held internally fixed |
| paper training contributions separated from thesis schedule | **PASS** | MuSGD/Progressive Loss/STAL = paper methodology; thesis uses frozen repo schedule |
| coffee schedule frozen | PASS | 50 epochs max, 640, batch16, workers2, patience15, optimizer auto, etc. |

### Critical guardrail

The proposal must **not** say that the thesis trains with MuSGD merely because MuSGD is a YOLO26 paper contribution. The experiment protocol is the repository configuration actually used.

---

## 4. Optimization claim audit

The title contains **Analisis dan Optimasi**. The methodology now operationalizes both words.

### Optimization genealogy

\[
\text{AF2 reference}
\rightarrow
\text{one-factor-at-a-time factorization}
\rightarrow
\text{validation screening}
\rightarrow
\text{limited sensitivity}
\rightarrow
\text{AF2* selection}
\rightarrow
\text{method freeze}
\rightarrow
\text{direct confirmation}.
\]

| Check | Verdict |
|---|---|
| optimization is more than hyperparameter rhetoric | PASS |
| candidates change one structural factor at a time | PASS |
| AF2WIN/ORI/POL/SOFT/LUM treated as factorization | PASS |
| PCG1/WAV1 not mislabeled as AF2 toggles | PASS |
| validation-only selection separated from locked test | PASS |
| method freeze precedes final interpretation | PASS |
| parent non-additivity used only as design rationale | PASS |

The parent-paper Table 6 evidence is used narrowly: AFAB-1 and AFAB-2 are not automatically additive on the aircraft benchmarks. It does not prove the same interaction on coffee.

---

## 5. Experiment genealogy audit

This is the highest-priority causal boundary in the proposal.

\[
\boxed{
\text{historical D0-parent factorization}
\neq
\text{official-pretrained direct confirmation}
}
\]

| Evidence family | Source initialization | Role |
|---|---|---|
| historical factorization | seed-matched coffee-trained D0 parent | mechanism/design genealogy |
| direct confirmatory comparison | same official `yolo26n.pt` artifact | thesis effectiveness comparison |
| seed-42 direct result | official-pretrained matched pair | preliminary/pilot feasibility evidence |
| future repeated-seed synthesis | same direct protocol | final thesis inference when experiment phase resumes |

Verdict: **PASS**. The current research-flow figures explicitly separate these paths.

Forbidden final diagram:

```text
D0 -> historical AF2 factorization -> final thesis model
```

Required final logic:

```text
historical factorization -> informs AF2* choice
                         -> method freeze
official yolo26n.pt -----+-> matched Native vs AF2* confirmation
```

---

## 6. Fairness and test-lock audit

Direct protocol requires:

- exact same official source checkpoint;
- frozen checkpoint hash;
- matched 21-class target-head initialization;
- exact persistent detector-state equality before training;
- equal detector parameter count;
- AF2 adds zero learned parameters;
- same schedule and seed per pair;
- validation used for development;
- locked test not used for optimization or model selection.

Verdict: **PASS**.

The statement “only intended treatment difference is deterministic input preprocessing” is supported at the study-protocol level.

---

## 7. Research-question coverage

| RQ | Bab III treatment | Verdict |
|---|---|---|
| RQ1 — optimize AF2 design | factorization + sensitivity + AF2* selection | PASS |
| RQ2 — native vs selected AF2* | matched direct confirmation | PASS |
| RQ3 — difficult/lower-tail classes | Bottom-3, Worst, per-class AP, paired errors | PASS |
| RQ4 — discrimination vs localization pattern | proposal accessibility + localization-conditioned Top-1 + correct-decision recall | PASS |

No RQ requires claiming that localization is solved.

---

## 8. Metric audit

Planned metric families are coherent:

- mAP50;
- mAP50–95;
- Macro mAP50–95;
- per-class AP;
- Bottom-3;
- Worst-class;
- raw top-500 proposal accessibility;
- localization-conditioned Top-1;
- correct-decision recall;
- parameters;
- latency;
- throughput;
- peak VRAM.

Verdict: **PASS**, with this interpretation rule:

> mechanism diagnostics are post-hoc evidence about where the performance change appears; they are not causal proof that one subsystem alone caused the gain.

---

## 9. Visualization and error-analysis audit

Figures/specifications now cover:

1. research framework;
2. Native vs AF2–YOLO26 architecture;
3. AF2 operator;
4. factorized optimization and method freeze.

Qualitative analysis is paired with deterministic/fixed-seed sampling and error categories:

- rescue: Native wrong → AF2 correct;
- regression: Native correct → AF2 wrong;
- stable-correct;
- unresolved/stable-error as applicable.

Verdict: **PASS**.

CAM/EigenCAM is not treated as mandatory until compatibility with YOLO26 is verified, preventing an unsupported visualization promise.

---

## 10. Efficiency audit

Correct formulation:

\[
\text{parameter-free} \neq \text{compute-free}.
\]

Since AF2 adds no learned frontend parameters but performs patch extraction, FFT, angular aggregation, inverse FFT, and reconstruction, runtime/memory measurements remain necessary.

Verdict: **PASS**.

Avoid describing AF2 as “lightweight” solely from parameter count.

---

## 11. Pilot-evidence audit

The completed direct seed-42 experiment may be used only as **preliminary feasibility evidence**.

Allowed:

> Studi pendahuluan pada seed 42 menunjukkan arah peningkatan pada aggregate dan lower-tail metrics di bawah protokol matched direct-from-pretrained.

Not allowed:

> AF2 terbukti unggul secara final pada penelitian ini.

Final repeated-seed inference is not yet available and is intentionally outside the current proposal-writing milestone.

Verdict: **PASS**.

---

## 12. Formal-assembly checklist

Before producing the campus document:

1. use `05_METHODOLOGY.md` as Bab III base;
2. replace embedded §3.5 with `05_05_AF2_PRIMARY_SOURCE_HARDENED.md`;
3. apply the locator correction in `BAB3_AF2_PAGE_LOCATOR_RECERTIFICATION_2026-08-25.md`;
4. redraw Figures 3.1–3.4 from `06_RESEARCH_FLOW.md`;
5. resolve citation keys through `CANONICAL_SOURCE_KEYS.md`;
6. preserve `[DET-01]` as preprint status;
7. label direct seed-42 result as preliminary;
8. do not expose locked test as a development source;
9. proofread equation numbering and page citations against final PDF/DOCX rendering;
10. run the cross-chapter audit after assembly.

---

## 13. Final disposition

Scientific/methodological proposal readiness:

\[
\boxed{\text{PASS}}
\]

Remaining work is document assembly and presentation QA, not a missing methodological argument and not a request for additional training experiments.