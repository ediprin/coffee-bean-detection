# Faruq-v3 Label Visual Audit Result

Date: 2026-08-02

Protocol: `faruq-v3-label-visual-audit-v1`

Selection: deterministic normalized-area quantiles

Training executed: no

Inference executed: no

Test images accessed: no

## Decision

**MIXED LABEL OBSERVABILITY — DOMAIN-EXPERT CONFIRMATION REQUIRED.** The visual
audit supports the earlier `DATA_OR_SCALE_LIMITED` decision, but it does not
show that the entire dataset is unusable. It shows that the flat 21-class
formulation mixes several different prediction problems whose evidence is not
equally observable from an uncalibrated RGB scene.

This review inspected six size-family sheets and twelve train/validation local
confusion-pair sheets. It is an AI-assisted visual review, not a replacement
for an SNI domain expert.

## Size-family findings

- `kulit_kopi` shows a broadly ordered apparent-area signal in train and
  validation, consistent with its strong numerical order AUROC. Distribution
  tails still overlap. SNI defines these labels by the fraction of an intact
  pericarp, not by absolute image-space area.
- `tanah_batu_ranting` also shows a useful ordered signal, although the class
  combines heterogeneous materials (soil, stone, and twig) whose shapes differ
  substantially.
- `kulit_tanduk` small, medium, and large overlap strongly in both splits.
  Several large examples appear smaller in full-frame context than medium
  examples, matching the non-monotonic medians and approximately 0.57 order
  AUROC. This does not by itself contradict SNI: the standard defines these
  labels by the fraction of an intact parchment (`<1/2`, `1/2-3/4`, `>3/4`),
  so fragmentation and shape are the relevant evidence rather than absolute
  box area. The contact sheets still show that this relative completeness is
  not applied with an obvious, stable visual boundary.

Consequently, a single RGB box-geometry head for every size family is not
justified. Coffee-husk and parchment classes need a relative-completeness or
fragmentation target. Only foreign matter uses physical millimetre thresholds
and therefore requires camera calibration/reference geometry. Merging size
labels is only permissible after confirming the operational SNI reporting
requirement.

## Local-defect findings

- `biji_muda` and `biji_bertutul_tutul` have substantial color and texture
  overlap in train and validation. Some examples have no crisp visual boundary
  between the two labels.
- `biji_hitam` and `biji_hitam_sebagian` form a severity continuum. Several
  partial-black examples are nearly as dark as full-black examples, so this is
  better represented as defect type plus extent/severity than as unrelated
  flat classes.
- One-hole and multiple-hole beans are often distinguishable in the zoomed
  crop, but the cue is tiny in full-frame context and can be hidden by pose.
  Their confusion with `biji_bertutul_tutul` is therefore partly a
  detail-resolution problem and partly a count/visibility problem.
- `biji_coklat` is generally more separable by global color, but mottled brown
  samples overlap the spotted class.

These observations explain why a refinement can improve proposal access yet
degrade class accuracy: the bottleneck is not one missing convolutional block.
It combines weak physical-scale observability, continuous severity boundaries,
tiny local cues, and heterogeneous label semantics.

## Consequence for the research design

Do not start another architecture search on the current flat label space. The
next scientific step is a domain-reviewed ontology specification that separates:

1. categorical object or defect type;
2. physical size, when supported by calibrated geometry;
3. severity or affected-area extent;
4. count attributes such as one versus multiple holes.

The corrected formal mapping is frozen in
`configs/sni21/structured_ontology_v1.yaml` and documented in
`docs/SNI21_STRUCTURED_TARGET_PROTOCOL.md`.

The original 21 labels and SNI scoring must remain recoverable from these
outputs. This is not authorization to silently merge or relabel classes. After
the ontology and observability rules are frozen, a controlled baseline can
compare the flat head with the structured formulation.
