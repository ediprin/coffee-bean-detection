# Faruq-v3 SFRNet C/SC-Former Breadth Screening Protocol

## Purpose

Screen two classification-only transfers from Cheng et al. (TGRS 2023) without modifying native YOLO26 box regression:

- **C1**: C-Former only.
- **SC1**: parallel S-Former + C-Former.

This is breadth discovery on seed 42, not confirmation and not a literal reproduction of RoI-based SFRNet.

## Paper-derived mechanism

SFRNet receives RoI features `F in R^(C x H x W)`, with `C=256`, `H=W=7` by default. Its SC-Former has two parallel transformer modules. The spatial module uses sinusoidal positional encoding and MSA over the `H*W` sequence. The channel module reshapes the feature into a channel sequence of length `C`, uses LSH-based sparse self-attention, derives Q and K from the same linear projection, and uses random-projection hashing to put nearby channel embeddings into buckets. The paper uses `B=4` buckets by default. Attention is performed within each bucket.

The paper reports that S-Former alone improves the baseline, C-Former alone does not provide a quantitative gain, while combining S-Former and C-Former is better than using either detached component alone. The paper gives no explicit algebraic equation for the final S/C feature-fusion operator; the parallel architecture is shown in Fig. 4.

## YOLO26 transfer boundary

YOLO26 has no 7x7 RoI classification tensor in its native one-stage Detect head. We therefore:

1. preserve native P3/P4/P5 box branches;
2. partition each dense classification field into non-overlapping 7x7 windows (padding only at field boundaries);
3. project each window to 64 channels;
4. for C1, transpose each window to 64 channel tokens of length 49;
5. apply shared-Q/K random-projection LSH ordering, split the sorted channel sequence into four equal buckets, and compute self-attention only within each bucket;
6. for SC1, run the existing transferred 7x7 S-Former in parallel and add spatial/channel refined fields;
7. map the refined field to a 21-class residual correction with a zero-initialized 1x1 classifier.

The additive S/C fusion in SC1 is a declared transfer choice because the source paper does not provide an explicit fusion equation. It must not be described as a literal SFRNet reproduction.

## Frozen settings

- seed: 42
- hidden channels: 64
- spatial heads: 4
- local window: 7x7
- LSH buckets: 4
- hash seed: 2023
- MLP ratio: 2.0
- correction scale: 1.0
- training schedule: same 50-epoch discovery budget as SF1
- evaluation: validation only
- test split: forbidden/unavailable

## Static gates

Before training, CI must verify:

- deterministic fixed hash projection;
- one shared Q/K projection (no separate query/key layers);
- every channel occurs exactly once in the LSH ordering;
- C1 zero-initialized correction reproduces native D0 scores and boxes;
- activating the correction can change class scores without changing boxes;
- SC1 contains both parallel spatial and channel blocks;
- gradients reach the C-Former shared-Q/K projection;
- notebook is branch-correct and val-only.

## Discovery decision

Each arm is evaluated independently against D0FT using the same broad-search criteria as SF1:

- macro mAP50-95 no worse than D0FT by more than 0.2 pp;
- bottom-3 no worse by more than 2 pp;
- worst-class no worse by more than 3 pp;
- at least one discovery signal: macro +0.2 pp, bottom-3 +0.5 pp, or worst +0.5 pp.

`RETAIN` at seed 42 is only authorization for later confirmation; it is not a thesis claim.
