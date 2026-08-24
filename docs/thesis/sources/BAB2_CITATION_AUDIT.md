# Bab II Citation Audit

Purpose: track reference breadth, citation recycling, source-key integrity, numerical verification, and promotion status for `docs/thesis/proposal/04_LITERATURE_REVIEW.md` plus its modular §2.9 companion `docs/thesis/proposal/04_09_RELATED_WORK_TABLE.md`.

Status: **FIRST-PASS SOURCE-GROUNDED BAB II COMPLETE**. Sections 2.1–2.8 are rewritten in the main Bab II file. Section 2.9 has been promoted in a modular companion file with an 18-study evidence-diverse table plus `Penelitian yang Diusulkan`. Remaining work is now chapter-wide normalization, not primary structural drafting.

## Current section status

| Section | Main source keys | Status | Remaining action |
|---|---|---|---|
| 2.1 Coffee beans / physical defects | STD-01, COF-07, COF-08, COF-02 | FUNCTIONALLY ADEQUATE | Final source-use normalization only. |
| 2.2 Conventional inspection | COF-07, COF-08, COF-14, COF-10, REV-01 | PASS FIRST REWRITE | COF-07/08 are currently over-reused across §2.1/2.2/2.6; remove them from §2.2 during final anti-recycling pass if equivalent direct evidence is available. |
| 2.3 Object detection | DET-02, DET-03, DIAG-01, DIAG-02, DIAG-03 | PASS FIRST REWRITE | Add evaluation source only where metric definition is needed. |
| 2.4 YOLO | DET-03, COF-06, COF-01, COF-02 | FUNCTIONALLY ADEQUATE | Preserve concise history; avoid adding more coffee detector papers here. |
| 2.5 YOLO26 | DET-01 | FUNCTIONALLY ADEQUATE | Repository protocol belongs primarily to Bab III. |
| 2.6 Fine-grained detection | FG-03, FG-02, COF-07, COF-08, COF-05, COF-04, COF-03, COF-12, COF-13 | PASS FIRST REWRITE | Keep FG-01 reserved for frequency-method bridge. |
| 2.7 Preprocessing | PRE-04, PRE-05, PRE-06, PRE-01, PRE-02, PRE-07, PRE-03, PRE-08 | PASS FIRST REWRITE | No major source gap. |
| 2.8.1 DFT/FFT | PRE-08, PRE-03, FG-01 | PASS PRIMARY-PAPER REWRITE / TEXTBOOK AUDIT OPEN | Pair final thesis with exact `THEORY-01`/`THEORY-02` edition/page once available. |
| 2.8.2 amplitude/phase | PRE-03, PRE-08, FG-01 | PASS FIRST REWRITE | Keep task-specific interpretation boundaries. |
| 2.8.3 radial/angular spectrum | SPEC-01, FG-01 | PASS FIRST REWRITE | SPEC-02 remains optional until its primary full text/page is re-opened. |
| 2.8.4 frequency-aware vision/detection | PRE-08, PRE-03, FG-01, FREQ-01, FREQ-02, FREQ-03, WAVE-01 | PASS FIRST REWRITE | Input-space vs feature-space separation already explicit. |
| 2.9 Related work | 18 prior studies + proposed study | PASS MODULAR FIRST REWRITE | Merge `04_09_RELATED_WORK_TABLE.md` into generated proposal; close the one explicitly marked PRE-05 quartile audit before final submission. |

## Current source-diversity count

Distinct citation keys used substantively across §2.1–§2.8 = **36** before counting the related-work table as additional use.

This meets the planning target of roughly 35–50 distinct authoritative/primary sources without requiring citation padding. The count is a coverage indicator, not a quality score.

Approximate route:

```text
coffee / standard / review       -> 13 distinct keys in domain sections
object detection / diagnostics   -> 6 distinct foundational/diagnostic keys
fine-grained theory              -> 2 dedicated general FG keys + coffee evidence
preprocessing                    -> 8 dedicated PRE keys
frequency / spectral             -> FG-01 + SPEC-01 + FREQ-01/02/03 + WAVE-01
```

## Anti-recycling finding

The large recycling problem present in the original scaffold has been substantially reduced:

- Hong is concentrated in §2.4 and the synthesis table;
- Hebert/Jundullah/Samudra are concentrated in §2.6;
- §2.7 has its own preprocessing corpus;
- §2.8 has its own Fourier/spectral corpus.

Two intentional-but-currently-overused cases remain:

- `COF-07` Kesiman appears substantively in §2.1 taxonomy, §2.2 manual-identification context, and §2.6 granularity difficulty;
- `COF-08` Arwatchananukul appears substantively in §2.1 taxonomy, §2.2 manual-sorting context, and §2.6 unseen-data fine-grained evidence.

Before final export, prefer deleting/replacing the §2.2 uses so both papers retain their stronger primary roles in §2.1 and §2.6. Do **not** add new papers merely to hide reuse; replacement evidence must genuinely support the manual-inspection sentence.

## §2.8 source-role audit

### Transform foundation

- `PRE-08` FDA provides a primary-paper 2-D DFT formulation, FFT statement, amplitude/phase decomposition, inverse transform, and non-learned Fourier input manipulation.
- `PRE-03` FE-YOLO independently provides complex Fourier coefficients and transform–process–reconstruct use before object detection.
- `FG-01` is used only to explain why its parent method uses patch-wise rather than global frequency responses for its fine-grained remote-sensing setting.
- `THEORY-01/THEORY-02` remain final-bibliography strengthening anchors; exact edition/page verification is still open.

### Amplitude and phase

- `PRE-03` directly supplies \(F=R+jI\), amplitude, and phase equations.
- `PRE-08` provides an independent example where low-frequency amplitude is changed while source phase is retained.
- `FG-01` supplies the parent-method choice to remodel amplitude while preserving phase during patch reconstruction.

Safe conclusion: amplitude and phase can be treated differently by Fourier-based methods. Unsafe conclusion: amplitude or phase is the proven coffee-defect bottleneck.

### Radial/angular representation

- `SPEC-01` Cao et al. is the main independent theory bridge: spectrum energy can be summarized through radial and angular distributions; radial behavior is used for frequency/periodicity/scale analysis and angular behavior for directionality.
- `FG-01` supplies the actual AFAB-2 angular-density formulation and entropy-conditioned suppression.
- `SPEC-02` is intentionally non-essential until its primary source is re-opened.

### Processing location

```text
INPUT / DATA SPACE
PRE-08 FDA       -> non-learned Fourier manipulation; segmentation/UDA
PRE-03 FE-YOLO   -> learned Fourier enhancement before YOLO
FG-01 AFAB       -> patch-wise data-space frequency processing for FGOD

FEATURE / NETWORK SPACE
FREQ-01 FFC      -> interconnected spatial/spectral processing
FREQ-02 FDADNet  -> spatial-frequency defect representation
FREQ-03 FDConv   -> adaptive frequency modulation in dense prediction
WAVE-01 WTConv   -> wavelet feature-processing precedent
```

This prevents the false claim that all frequency-aware methods are equivalent forms of high-pass preprocessing.

## §2.9 related-work table audit

`04_09_RELATED_WORK_TABLE.md` now contains 18 prior studies + the proposed study, deliberately balanced as:

- **9 coffee/fine-grained coffee studies**: Hong, Gope 2024, Bahy, Jundullah, Hebert, Samudra, Arwatchananukul, Jiao, Hu;
- **4 preprocessing studies**: IA-YOLO, Syauqi white pepper, Chen maize, FE-YOLO;
- **5 fine-grained/frequency studies**: Xu/AFAB, Xie/DRNet, FFC, FDADNet, FDConv;
- **1 proposed row**: AF2 + native YOLO26.

This is intentionally broader than the earlier 10-row table and avoids making the state-of-the-art section look like a coffee-YOLO-only survey followed by an unexplained spectral method.

Index-status discipline:

- verified Q1/Q2/SINTA labels from the index-audited master are used where available;
- conference rows are named by venue rather than assigned fictitious journal quartiles;
- `PRE-05` Computers and Electronics in Agriculture is explicitly marked **quartile audit open** in the working table rather than guessed;
- FE-YOLO is recorded as Q2 Digital Signal Processing using the 2024 SJR journal record; this external verification should be copied into the master index before final export.

## Parent-method boundary

`FG-01` Xu et al. remains the closest parent source, but its scope is constrained:

- LFDet contains AFAB, CGFI, and FTIF;
- AFAB contains patch-wise DFT, AFAB-1 adaptive high-pass filtering, and AFAB-2 angular-amplitude suppression;
- AFAB-2 must not inherit full LFDet headline gains;
- aircraft-domain evidence does not validate coffee transfer;
- coffee-specific residual reconstruction and integration are thesis implementation choices described in Bab III.

## Citation-key integrity

Never restore deprecated meanings:

```text
FREQ-01 = Cao 2019
FREQ-02 = Zhang & Tan 2003
FREQ-03 = WTConv
FREQ-04 = FDConv
```

Canonical namespace:

```text
SPEC-01 = Cao et al. 2019
SPEC-02 = Zhang & Tan 2003
WAVE-01 = WTConv
FREQ-01 = Fast Fourier Convolution
FREQ-02 = FDADNet
FREQ-03 = Frequency Dynamic Convolution
```

## Remaining gates before `FULL CITATION-READY`

- [x] Official coffee standard + independent taxonomy sources.
- [x] Foundational object-detection literature.
- [x] Original YOLO source.
- [x] YOLO26 primary source with preprint status stated.
- [x] General FG/FGOD + multiple independent coffee difficult-class sources.
- [x] Fixed, task-driven, transform-domain, agricultural, and Fourier preprocessing precedents.
- [x] Primary-paper DFT/amplitude/phase equations.
- [x] Radial/angular theory uses canonical `SPEC-*` namespace.
- [x] Input-space vs feature-space frequency methods separated.
- [x] §2.9 expanded to an evidence-diverse table with proposed-research row.
- [x] Distinct substantive source count >=35.
- [ ] Exact textbook edition/page audit for final DFT/FFT foundation.
- [ ] Reduce §2.2 reuse of COF-07/COF-08 where supported by equivalent primary evidence.
- [ ] Re-open every numerical claim once during final copyedit and record page/table pointers.
- [ ] Close all remaining table index/quartile fields, especially PRE-05.
- [ ] Merge modular §2.9 table into the generated proposal document and run final citation numbering/bibliography resolution.

## Current audit metrics

```text
unique_sources_total        = 36 substantive keys before table-only reuse
max_empirical_section_reuse = 3 (COF-07, COF-08; flagged for reduction)
uncited_technical_paragraph = no known major gaps in §2.1–§2.8; final copyedit required
unresolved_keys             = 0 in active canonical namespace
unverified_numeric_claims   = final page/table recertification pass pending
unverified_index_claims     = PRE-05 + final table recertification pending
```
