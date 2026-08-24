# Pre-Bab III Hong-Alignment Checkpoint

Date: 2026-08-25
Branch: `proposal/thesis-foundation`

## Purpose

This checkpoint preserves the proposal state **before** revising Bab III based on the full-text review of Hong et al. (2026).

No Bab III restructuring from the Hong audit has been applied in this checkpoint.

## State frozen at this checkpoint

- Bab I working draft and problem formulation already exist.
- Bab II first-pass structure, related-work table, reference allocation, and citation audits already exist.
- Bab III current draft exists in `docs/thesis/proposal/05_METHODOLOGY.md` and remains the pre-Hong-alignment version at this checkpoint.
- Research-flow draft exists in `docs/thesis/proposal/06_RESEARCH_FLOW.md`.
- Cross-chapter and protocol audits exist under `docs/thesis/sources/`.
- Direct AF2 pilot remains preliminary seed-42 evidence only.

## Hong audit findings not yet applied

The following findings have been discussed but are intentionally **not yet incorporated** into Bab III at this checkpoint:

1. Bab III should open with an overall architecture / research framework before operator equations, following the methodological sequencing seen in Hong.
2. Each proposed intervention should be explained using the pattern:
   `targeted problem -> exact insertion point -> mathematical operator -> expected role -> experiment that tests the claim`.
3. The word `Optimasi` in the thesis title requires an explicit AF2 optimization/sensitivity stage, not merely native YOLO26 vs AF2-YOLO26 comparison.
4. Candidate AF2 optimization factors should be method variables such as patch size, overlap, gamma, and angular-bin resolution; engineering parameters such as chunk size and eps should not be treated as methodological optimization factors.
5. `radius_ratio` is not an active factor for current `mode=af2`; it belongs to AF1/AF12 behavior in the implementation.
6. Optimization should be performed on development/validation data, followed by method freeze before any locked-test use.
7. Final confirmation should remain matched and repeated across seeds; test data must not become a tuning oracle.
8. Primary evaluation should remain centered on Macro mAP50-95 plus lower-tail metrics, with mAP50 as secondary context.
9. Mechanism wording must remain diagnostic and conservative: observed changes may be `consistent with` stronger class discrimination rather than being claimed as causal proof of localization/classification mechanisms.
10. Bab III should preserve grouped-split and leakage-control discipline even where Hong's written augmentation/split description is less explicit about parent grouping.

## Next revision target

The next Bab III revision will be performed from this checkpoint and should produce a clearly versioned Hong-aligned methodology structure without overwriting the historical pre-revision state.
