# Faruq-v3 IGEM Breadth Screening Protocol

## Purpose

Screen a classification-focused transfer of PCA-DB/IGEM on the 21-class coffee detector. This is seed-42 validation-only breadth discovery, not final confirmation.

## Retained source mechanism

The implemented branch retains the source-described structure already encoded in `coffee_detector.igem`:

- three 3x3 convolutions form the class-aware reference branch;
- a 1x1 mask head predicts `N+1` semantic channels including background;
- mask auxiliary loss weight is fixed to `0.05`;
- IGEM combines grouped static local context, dynamic local multi-head aggregation, and learned two-way channel fusion;
- enhanced representation produces a classification residual while the native box branch is preserved.

## Coffee/YOLO26 transfer boundary

Aircraft-specific fine cross masks are not valid for coffee beans. Training boxes are therefore rasterized as coarse axis-aligned class-aware rectangular masks. If boxes overlap, larger rectangles are painted first and smaller rectangles last. This overlap rule is a transfer choice.

The accessible source used for this implementation did not expose numerical defaults for local kernel size, head count, or channel reduction. Breadth settings are explicitly frozen as transfer choices:

- kernel size: 3
- attention heads: 4
- channel reduction: 4

The fine-class correction is zero initialized, so the candidate starts from the transferred native D0 detector logits. Box outputs remain native. The auxiliary mask loss is applied only to the one-to-many training branch.

## Frozen screening

- seed: 42
- epochs: 50
- image size: 640
- batch: 16
- D0 checkpoint initialization: mandatory
- evaluation split: validation only
- locked test: unavailable/forbidden

The branch runner emits Macro mAP50-95, bottom-3 class mAP50-95, and worst-class mAP50-95. The master breadth controller recomputes the common D0FT retention/discovery gate centrally.
