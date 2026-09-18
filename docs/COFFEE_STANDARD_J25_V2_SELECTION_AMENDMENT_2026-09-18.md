# Coffee Standard J25 v2 Selection Amendment

> **RETRACTED — FILENAME COMPONENTS ARE NOT SOURCE IDENTITIES.** Do not build
> or use J25 v2. Its 267 components and 253 selected images were derived from
> non-unique filename prefixes, not authoritative Roboflow asset IDs. The rule
> can merge unrelated photographs and quarantine valid data. The builder now
> fails fast. Use the author-provided official split and the thesis-provenance
> audit instead. The text below is historical only.

Status: **FROZEN DATA AMENDMENT — TRAINING NOT AUTHORIZED**  
Date: 2026-09-18

## Reason for the amendment

The J25 v1 visual audit inspected all three candidate splits and a deterministic
geometry sheet. Most displayed boxes covered plausible objects, but the source
sibling audit found 14 of 267 recoverable identities with different per-class
annotation-count signatures across generated siblings. Version 1 also preferred
the sibling containing the most boxes. That rule can select a crop or copy-paste
outlier and is therefore not retained.

## Frozen v2 rule

1. Identity components are still formed only from Roboflow parent IDs and exact
   hashes.
2. Per-class annotation counts are compared across every sibling in a component.
3. A component with more than one count signature is quarantined in full. No
   sibling from it can enter train, validation, or candidate test.
4. For every remaining annotation-consistent identity, one observed image is
   selected as the visual medoid. Distance is normalized dHash Hamming distance
   plus 0.25 times normalized maximum mean-RGB distance.
5. A deterministic hash resolves exact medoid ties.
6. Eligible components are split approximately 70/15/15 with the same seeded,
   class-aware optimizer used by v1.
7. All 25 labels must remain present in each split; parent/hash overlap must be
   zero; each target split fraction must remain within three percentage points.

The visual medoid is a source-selection rule, not a learned score. It does not
inspect a detector, validation metric, or test result.

## Required artifacts

- `coffee_standard_j25_v2_manifest.json`
- `coffee_standard_j25_v2_components.json`
- `coffee_standard_j25_v2_quarantine.json`
- `coffee_standard_j25_v2_summary.json`
- v2 class-review sheets and geometry sheet

## Authorization boundary

Passing the v2 technical gates does not authorize training. A fresh visual audit
of v2 and resolution of the public-source provenance discrepancy remain required.
The 451-image/25-class thesis description, 2,000-image/20-class paper description,
and 993-image public export must not be described as equivalent without evidence.
