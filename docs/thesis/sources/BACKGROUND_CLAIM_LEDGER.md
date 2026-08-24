# Background Claim Ledger

Purpose: make every important Bab I claim traceable to evidence and prevent unsupported prose expansion.

Status labels:

- **PAPER** — directly supported by one or more primary papers.
- **SYNTHESIS** — cross-paper interpretation allowed only within the stated scope.
- **HYPOTHESIS** — proposition to be tested by this thesis.
- **PILOT** — repository experiment evidence; preliminary only.

| Claim ID | Type | Proposal claim | Evidence keys | Safe wording | Verification before final proposal |
|---|---|---|---|---|---|
| BG-01 | PAPER | Manual visual coffee-defect inspection motivates automation | COF-01, STD-01 | Manual/visual inspection has subjectivity, consistency and throughput limitations described in the coffee literature | Reopen Hong introduction and SNI scope; avoid adding unsupported labor/time statistics |
| BG-02 | SYNTHESIS | Coffee inspection research has shifted toward learned representations and YOLO-family detection | COF-01, COF-06, COF-12 | Recent coffee studies increasingly use CNN/Transformer/YOLO methods | Do not claim a complete historical chronology without broader review |
| BG-03 | PAPER | YOLO is viable for green-coffee object detection | COF-01, COF-06, COF-16 | Multiple coffee studies report strong YOLO-family performance in their own datasets | Keep each result scoped to its taxonomy and dataset |
| BG-04 | SYNTHESIS | Strong few/coarse-class performance does not establish fine-grained multi-class resolution | COF-06, COF-07, COF-16 | Results across different studies suggest that taxonomy granularity matters | Do not compare absolute metrics across incompatible tasks as if directly paired |
| BG-05 | PAPER | Moving from 3 coarse classes to 17 specific defect classes is substantially harder in the Kesiman benchmark | COF-07 | State exact 3-class and 17-class accuracies as paper-scoped classification results | Verify table/page in primary PDF before final prose |
| BG-06 | PAPER | 17-class coffee classification can degrade on unseen data | COF-08 | Report the paper's controlled CV and unseen result separately | Verify exact protocol and page/table |
| BG-07 | PAPER | Visually similar coffee defect categories are directly reported as difficult | COF-01, COF-03, COF-04, COF-05, COF-13 | Authors report confusion/difficulty for visually close or subtle categories | Do not introduce frequency causality |
| BG-08 | SYNTHESIS | Fine-grained coffee detection is increasingly a representation/discrimination problem, not only an object-presence problem | COF-03, COF-04, COF-05, COF-07, COF-13 | "The audited literature indicates..." | Keep as synthesis; do not write as universal law |
| BG-09 | SYNTHESIS | Current coffee solutions predominantly modify internal representation | COF-01, COF-12, COF-13 plus supporting coffee atlas | "Within the audited coffee corpus, many recent methods..." | Avoid global "all prior work" wording |
| BG-10 | PAPER | Detection-oriented preprocessing can outperform generic visual enhancement | PRE-01, PRE-02 | IA-YOLO/DENet explicitly optimize enhancement for downstream detection | Keep adverse-weather domain visible |
| BG-11 | PAPER | Fourier-domain enhancement can be placed before YOLO | PRE-03 | FE-YOLO processes Fourier amplitude/phase before detection | Do not describe FE-YOLO as parameter-free or angular |
| BG-12 | PAPER | Fixed preprocessing before YOLO has agricultural seed/spice precedents | PRE-04, PRE-05 | White pepper and maize studies evaluate preprocessing before YOLO | Keep preprocessing pipelines and task scopes accurate |
| BG-13 | PAPER | Angular Fourier-energy distributions can describe directional texture information | FREQ-01, FREQ-02 | Use as theoretical support for frequency-angular terminology | Do not claim coffee-specific discriminative effectiveness |
| BG-14 | PAPER | Frequency-aware processing has fine-grained detection precedent | FG-01, FG-02 | Fine-grained aircraft/remote-sensing work uses discriminative/frequency representations | Transfer to coffee remains unvalidated |
| BG-15 | HYPOTHESIS | AF2 may improve fine-grained coffee discrimination without materially improving raw localization accessibility | COF problem evidence + method bridge + PILOT-01 | Phrase as research hypothesis / question | Must remain hypothesis until repeated validation |
| BG-16 | PAPER/SYNTHESIS | Aggregate metrics can conceal weak classes | COF-02, COF-04, COF-05 | Use class-wise evidence to motivate Bottom-3/Worst-class reporting | Do not imply Bottom-3 is a standard metric from those papers; it is our analysis choice |
| BG-17 | PAPER | Classification confidence and localization quality are not equivalent | DIAG-01, DIAG-02, DIAG-03, COF-09 | Supports separate interpretation of classification and localization effects | Preserve architecture/task differences among sources |
| BG-18 | PILOT | AF2-direct seed 42 shows positive macro and lower-tail deltas with unchanged raw proposal accessibility | PILOT-01 | Always call this preliminary / feasibility evidence | Never convert to final superiority claim until repeated-seed validation |

## High-risk words requiring review

Whenever the following words appear in proposal prose, re-check this ledger:

- "menyebabkan"
- "terbukti"
- "pertama"
- "belum pernah"
- "SOTA"
- "mengatasi"
- "secara signifikan"
- "robust"
- "optimal"
- "efisien"

Use stronger causal or novelty language only when the corresponding evidence and statistical protocol support it.

## Claim-generation rule

A future document generator should follow:

```text
claim needed
   ↓
find Claim ID
   ↓
read safe wording + scope
   ↓
open primary PDF if numerical/technical
   ↓
draft sentence
   ↓
attach citation
```

If no Claim ID exists, add and verify the claim here before expanding the proposal narrative.