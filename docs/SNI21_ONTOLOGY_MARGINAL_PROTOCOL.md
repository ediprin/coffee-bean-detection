# SNI-21 Ontology-Marginalized Classification Protocol

Status: frozen implementation protocol; training not authorized

## Research question

Can semantic supervision derived from the SNI-21 ontology improve the
classification component of YOLO26 without changing localization, adding an
inference head, or changing the official 21-class output?

## Models

- `D0`: unchanged YOLO26n P3-P5 baseline.
- `C0`: the same detector with identity-control auxiliary loss.
- `S0`: the same detector with ontology-marginalized auxiliary loss.

All models use the same initialization, image size, augmentation, optimizer,
schedule, seed, and grouped Faruq-v3 train/validation split. `C0` and `S0` have
identical task masks and weights. Neither adds trainable parameters or changes
the inference graph.

## Loss

For leaf logits `z` and leaf probability `p(c|x)`, the probability of ontology
value `v` for task `t` is:

```text
p_t(v|x) = sum_{c: mapping_t(c)=v} p(c|x)
```

`S0` applies negative log likelihood to the mapped ontology value. `C0` uses
the same per-task applicability mask and weight but repeats leaf-class cross
entropy. The total task loss is a weighted mean, then multiplied by the frozen
gain `0.20` and added to YOLO's classification loss.

Included tasks:

- entity family;
- primary condition;
- hole count;
- integrity fraction;
- surface extent.

Excluded tasks:

- physical size: blocked without calibrated scale;
- relative completeness: pending domain-expert review;
- positive flags: positive-only partial supervision is not converted to
  arbitrary negative labels.

## Screening and gates

Run validation seed 42 only after the static implementation audit passes.
Compare `S0` against both `D0` and `C0`.

Required gates:

- Macro mAP50-95 improves by at least 0.5 point over both controls;
- conditional top-1 class accuracy improves by at least 2 points over `D0`;
- bottom-3 mAP50-95 drops no more than 1 point;
- worst-class mAP50-95 drops no more than 2 points;
- proposal accessibility drops no more than 1 point;
- inference parameter count and latency remain equivalent to `D0` within
  measurement tolerance.

Failure against either `D0` or `C0` stops the method. Three seeds and test stay
locked until every validation gate passes. No hyperparameter search is allowed
after reading validation results; `auxiliary_gain=0.20` is frozen for v1.
