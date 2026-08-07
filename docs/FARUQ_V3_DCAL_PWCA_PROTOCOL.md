# Faruq-v3 DCAL PWCA Breadth Screening

This experiment transfers the PWCA mechanism from Zhu et al. (CVPR 2022) to YOLO26. It is not a literal reproduction of the paper's DeiT/ViT architecture.

## Source mechanism

For target image I1, the paper computes pair-wise cross-attention using query from I1 and concatenated key/value from I1 and a randomly paired image I2:

`Kc=[K1;K2]`, `Vc=[V1;V2]`,

`PWCA(Q1,Kc,Vc)=softmax(Q1 Kc^T/sqrt(d)) Vc`.

The paired image regularizes attention learning. PWCA uses the same target as the main branch and is removed at inference.

## YOLO26 transfer

Only the P5 one-to-many classification training path is modified. P5 is projected to hidden tokens and processed by one attention block plus FFN. A zero-initialized 1x1 classifier produces a residual leaf-logit correction.

Unchanged: P3/P4 classification, all box branches, TAL assignment, one-to-one branch, and inference.

At evaluation/inference the wrapper delegates directly to native YOLO26 Detect, so the PWCA module is not executed.

## Arms

- `SA0`: P5 self-attention control. Query/key/value come from the same image.
- `PW1`: capacity-matched pair-wise cross-attention. Query comes from the target image; key/value concatenate target and paired-image tokens.

Primary mechanistic attribution is `PW1 - SA0`.

## Transfer choices

These are experiment choices, not paper hyperparameters: P5 only, one block, hidden dimension 64, four heads, FFN ratio 2, random non-zero cyclic in-batch pairing, and zero-initialized residual correction. Batch size one falls back to self-attention.

GLCA is excluded because the paper selects local queries using accumulated Transformer attention rollout, for which native YOLO26 has no equivalent source.

## Frozen breadth protocol

Seed 42, 50 epochs, image size 640, batch 16, D0 initialization, D0FT primary control, ACMC1 selected-model reference, validation only, test locked.

An arm is retained only if macro drops no more than 1.0 pp, bottom-3 and worst drop no more than 2.0 pp, and at least one signal appears: macro +0.2 pp, bottom-3 +0.5 pp, or worst +0.5 pp.

`RETAIN` authorizes later confirmation only; it does not authorize test evaluation.
