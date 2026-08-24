# Thesis Proposal Changelog

## v0.2.0 — 2026-08-24

Evidence-to-draft layer added.

### Added

- `sources/COFFEE_EVIDENCE_MATRIX.md`: core coffee-domain problem evidence and overclaim boundaries;
- `sources/METHOD_BRIDGE_MATRIX.md`: preprocessing, frequency, angular, fine-grained, agricultural, and classification/localization bridge literature;
- `sources/BACKGROUND_CLAIM_LEDGER.md`: claim-by-claim traceability and safe wording for Bab I;
- `proposal/01_PROPOSAL_SKELETON.md`: Bab I–III structure, RQs, objectives, boundaries, metrics and fair-comparison rules;
- `proposal/02_BACKGROUND.md`: first evidence-grounded working draft of the thesis background.

### Strengthened rules

- keep coffee-domain evidence and non-coffee mechanism evidence as separate layers;
- every numerical claim must be re-opened in the primary PDF before final proposal prose;
- cross-paper synthesis must be labeled as synthesis rather than a fact from one source;
- terms such as `terbukti`, `menyebabkan`, `pertama`, `SOTA`, `optimal`, and `signifikan` require explicit evidence review;
- Bottom-3/Worst-class are our analysis choices motivated by class-wise heterogeneity, not metrics claimed as standard by the coffee papers;
- AF2-direct seed-42 remains pilot evidence only.

## v0.1.0 — 2026-08-24

Initial proposal-foundation freeze.

### Added

- dedicated branch `proposal/thesis-foundation` based on `codex/af2-direct-from-pretrained`;
- frozen thesis reasoning chain;
- Hong-centered literature-treatment method;
- evidence/claim guardrails;
- three-layer research gap;
- research questions, objectives, and scope boundaries;
- AF2-direct seed-42 pilot evidence with preliminary-only wording;
- document-generation guide;
- proposal workspace scaffold;
- source-registry guidance.

### Frozen decisions

- proposal narrative is problem-driven, not experiment-genealogy-driven;
- AF2 is framed as parameter-free input-space frequency-angular preprocessing;
- fine-grained discrimination is the primary problem framing supported by coffee literature;
- frequency is a candidate solution space, not a proven coffee-domain bottleneck;
- classification and localization effects must be diagnosed separately where possible;
- tail metrics must accompany aggregate performance;
- no further exploratory module stacking is required before proposal drafting begins.

### Not frozen as final thesis conclusions

- AF2 superiority across seeds;
- global novelty/"first" claims;
- final choice of optional CLAHE or operator ablations;
- final wording of "Analisis dan Optimasi" in the title.

Future conceptual changes should be recorded here and committed before dependent proposal chapters are revised.
