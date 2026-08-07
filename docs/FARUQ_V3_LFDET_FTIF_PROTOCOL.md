# Faruq-v3 LFDet FTIF Breadth Screening Protocol

## Purpose

Evaluate the language-signal mechanism of Xu et al. (Neural Networks, 2025) independently from AFAB and CGFI on the controlled 21-class coffee benchmark.

Predeclared arms:

- **FT1** — specific prompt only + text-to-image cross-attention, no alignment loss.
- **FT2** — base + specific prompt embeddings + cross-attention, no alignment loss.
- **FT3** — base + specific prompt embeddings + cross-attention + bidirectional alignment.

This is seed-42 breadth discovery, not confirmation.

## Paper-derived FTIF mechanism

The LFDet paper defines FTIF as three parts: fine-grained text prompt generation, text-to-image feature integration, and bidirectional image-text alignment.

### Fine-grained prompts

The paper uses hand-crafted prompts and separates shared information into a **base prompt** and distinguishing class information into a **specific prompt**. A frozen CLIP text encoder produces both embeddings, and the two embeddings are added to form the final class text representation. The text encoder is not fine-tuned with the detector.

### Text-to-image integration

For visual token matrix `I_e` and text matrix `T_e`, the paper first projects text to the visual dimension `T*_e`. The visual tokens are queries; text tokens are keys and values:

`Q_I = I_e W_Q`, `K_T = T*_e W_K`, `V_T = T*_e W_V`  (Eq. 15)

`CrossAttn = softmax(Q_I K_T^T / sqrt(d)) V_T`  (Eq. 16)

`I_MCA = I_e + CrossAttn`  (Eq. 17)

followed by the paper's projection/GELU/projection residual form (Eq. 18).

### Bidirectional alignment

The paper computes cosine similarity between the pre-interaction visual token and projected text token:

`S = cosine(I_e, T*_e) / tau`  (Eq. 19)

with `tau = 0.07`.

Positive visual-to-text alignment uses softmax cross-entropy; background/negative visual tokens use sigmoid BCE because no fine-grained text corresponds to background. Text-to-image alignment is formulated as binary sigmoid/BCE. Eq. (20) combines positive and negative losses from both directions:

`L_bi = 0.5(L+_I2T + L+_T2I) + 0.5(L-_I2T + L-_T2I)`.

The paper reports that one-direction alignment can degrade performance, whereas bidirectional alignment is more stable.

## Coffee/SNI prompt construction

All prompts are frozen in `configs/ftif/sni21_prompts.yaml` and are derived only from `configs/sni21/structured_ontology_v1.yaml`.

Forbidden sources for prompt construction:

- validation confusion matrices;
- validation hard-class rankings;
- residual-error audit pairings;
- test images or labels.

The shared base prompt describes one item in a green-coffee quality inspection sample. Specific prompts express ontology-level attributes such as black/partial-black appearance, hole count, broken integrity, parchment/husk family, and relative-completeness categories.

The SNI physical-size foreign-matter categories are explicitly documented as **calibrated-scale-required** in the ontology. Their millimeter wording is preserved only as semantic prior text; the experiment must not claim that an uncalibrated RGB image directly reveals metric size.

## Frozen text encoder

The paper states that a pretrained CLIP text encoder is frozen, but the paper text available to this project does not identify a concrete CLIP model variant. This transfer therefore freezes:

- library: `open_clip_torch`;
- model: `ViT-B-32`;
- pretrained weights: `openai`;
- text encoder: frozen, embedding cache generated once before training.

This model choice is a transfer choice, not a paper-reproduction claim.

Base and specific embeddings are encoded separately without normalization and directly added for FT2/FT3 because the paper states that the embeddings are added. Cosine normalization is applied only when constructing Eq. (19).

## YOLO26 transfer contract

The original LFDet replaces the fine-grained classification head with FTIF. To isolate the language mechanism while retaining the already-trained YOLO26 detector, this experiment uses an identity-start residual transfer:

1. Native YOLO26 box heads receive untouched P3/P4/P5 features.
2. Native YOLO26 leaf classification logits are preserved.
3. FTIF computes paper-style text-to-image representations on each P3/P4/P5 level.
4. A zero-initialized 1x1 correction maps the FTIF representation to 21 leaf logits.
5. Final classification logits are `native_logits + FTIF_correction`.

Thus at initialization candidate outputs exactly reproduce native D0 while still allowing FTIF to learn a classification correction. This stabilization is a deliberate YOLO26 transfer choice and must not be described as the literal LFDet head replacement.

FTIF is active at inference. Unlike APCL/MRL, it is not training-only.

## Alignment transfer to YOLO26

FT3 uses native YOLO26 one-to-many Task-Aligned-Assigner positives as the paper's dense classification label space. The assigner itself is not modified.

- similarity matrices are computed only for one-to-many training predictions;
- one-to-one training retains the FTIF classification representation but does not receive the auxiliary alignment loss;
- positive and negative terms are mean-reduced separately before Eq. (20), preventing the number of dense background tokens from silently changing the loss coefficient;
- alignment weight is frozen to `1.0` because an explicit balancing coefficient for `L_bi` was not identified in the paper text used here. This is a transfer choice.

## Frozen breadth settings

- seed: 42
- epochs: 50
- image size: 640
- batch: 16
- prompt manifest: frozen SNI-21 manifest
- temperature: 0.07
- alignment weight: 1.0
- D0 checkpoint initialization: mandatory
- evaluation: validation only
- test: unavailable/forbidden

## Discovery decision rule

Each FTIF arm is compared with D0FT:

- macro mAP50-95 no worse by more than 0.2 pp;
- bottom-3 no worse by more than 2 pp;
- worst-class no worse by more than 3 pp;
- at least one discovery signal: macro +0.2 pp, bottom-3 +0.5 pp, or worst +0.5 pp.

`RETAIN` only authorizes later confirmation. It is not final evidence.
