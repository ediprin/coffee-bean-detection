# Coffee Standard J25 SAFE-D0 Causal Control

Status: **frozen before training**

Date: 2026-09-19

## Question

Did the aggregate improvement of `AF2LUMSAFE` come from its semantic-safe
training policy, or from stochastic AF2 exposure during training?

## Single controlled arm

`SAFED0` is trained fresh from the same official `yolo26n.pt` initialization
as `AF2LUMSAFE`. It uses the exact same J25 train-siblings-v2 development
split, seed 42, 50-epoch schedule, semantic-safe augmentation, source-identity
repeat-factor sampler, and native YOLO26n P3 detector. It has no input
frontend and never applies AF2 during training or inference.

The static audit must establish exact equality of the initialized native
detector state, training dictionary, sampler dictionary, model YAML, and
parameter count against the detector inside `AF2LUMSAFE`. The only intended
difference is stochastic AF2 exposure.

## Evaluation

The completed `SAFED0` checkpoint is evaluated once on the unchanged
validation split. It is compared with:

- matched historical `D0DIRECT`, which isolates the full safe-policy effect;
- completed `AF2LUMSAFE` at inference strength zero, which isolates the
  training-time stochastic-AF2 contribution while both endpoints receive raw
  RGB at inference.

Report Macro, Bottom-3, Worst-class, and AP for `Biji Hitam Pecah`. The result
is descriptive single-seed causal evidence. No extra seed or test evaluation
is authorized by this protocol. The locked test is neither extracted nor
opened.

