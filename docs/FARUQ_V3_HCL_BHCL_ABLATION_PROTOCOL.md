# Faruq-v3 Paired HCL1 vs BH1 Breadth Ablation

## Why this control is necessary

The BHCL paper separates ordinary hierarchical contrastive learning (HCL) from the prototype-based balanced variant (BHCL). Therefore a BHCL-only coffee experiment cannot attribute any gain to class balancing/prototypes: it could simply be caused by hierarchical positives.

This protocol runs two paired arms on exactly the same YOLO26 scaffold:

- **HCL1:** Eq. (6)-(7) hierarchical supervised contrastive learning, no prototypes and no class-balanced denominator.
- **BH1:** the same hierarchy/projection plus Eq. (8)-(10) prototype-based class balancing.

## Shared contract

Both arms use:

- the same D0 seed-42 checkpoint;
- the same native one-to-many TAL foreground assignments;
- the same 128D P3/P4/P5 projection;
- the same SNI-21 strict two-level hierarchy (`entity_family -> leaf`), root excluded;
- `tau=0.1`;
- auxiliary weight 0.6;
- the same 50-epoch project-controlled training schedule, image size 640, batch 16;
- native one-to-one loss unchanged;
- no auxiliary projection during inference;
- validation only, with locked test unavailable.

Using weight 0.6 for HCL1 is a controlled attribution choice: the paper reports `lambda_BHCL=0.6`; matching the auxiliary scale makes the HCL1→BH1 difference focus on balancing rather than loss magnitude.

## HCL1

Eq. (6) uses all foreground representations except anchor `i` in its denominator. At hierarchy level `l`, positives are other foreground representations sharing the same ancestor category. Level penalties follow Eq. (7).

The source paper generates two augmented views, so positive sets are expected to exist. This YOLO transfer does not create explicit paired views; native TAL produces multiple positive locations for matched objects. As a defensive rule only, if an anchor has no non-self positive at a level, that undefined level term contributes zero for that anchor while the outer normalization remains the number of foreground anchors.

HCL1 never updates or reads class prototypes.

## BH1-only difference

BH1 adds:

1. a prototype for every non-root category node;
2. Eq. (8) category-balanced denominator, where each category contributes its average similarity rather than scaling with raw mini-batch frequency;
3. the own ancestor prototype as a positive;
4. Eq. (10) EMA prototype updates with `epsilon=0.1`.

Thus `BH1 - HCL1` is the clean internal estimate of the prototype/class-balancing contribution under this YOLO26 transfer.

## Source-equation sign note

Eq. (6)/(8) already define pair losses with `-log`, while Eq. (7)/(9) display an additional outer minus. Both arms use the positive aggregation of the defined `-log` pair losses; applying both printed minus signs literally would reverse the optimization objective.

## Discovery outputs

For D0FT, ACMC1, HCL1 and BH1 report:

- macro mAP50-95;
- bottom-3 class mAP50-95;
- worst-class mAP50-95.

Also report `BH1_minus_HCL1` for all three metrics. Each arm independently receives `RETAIN/REJECT` under the shared breadth-search gate. No seed-42 result is a final thesis claim.
