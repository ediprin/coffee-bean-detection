# SNI-21 Structured Target Protocol

Version: 1.0.0

Status: ontology specification only; training is not authorized

## Evidence and correction

This protocol is based on the definitions and defect values in SNI
01-2907-2008, not on class names alone. It corrects an earlier over-generalized
interpretation of the size labels:

- coffee husk and parchment sizes are fractions of the corresponding intact
  covering (`<1/2`, `1/2-3/4`, `>3/4`);
- foreign matter sizes use physical length or diameter (`<5`, `5-10`, `>10
  mm`);
- black versus partial black is defined by affected surface extent;
- one versus multiple holes is an explicit count attribute;
- black-broken is a compound condition, not an unrelated visual category.

Therefore normalized box area is only a diagnostic proxy. It is not the SNI
definition for coffee husk or parchment size, and it cannot recover physical
millimetres for foreign matter without calibrated scale.

## Target factorization

The frozen mapping in `configs/sni21/structured_ontology_v1.yaml` retains every
original SNI-21 leaf and defect weight while representing its semantics as:

1. entity family;
2. primary condition;
3. positive defect flags;
4. affected surface extent;
5. integrity fraction;
6. relative completeness of a covering;
7. hole count;
8. calibrated physical-size bin when applicable.

Unspecified attributes are unknown, not negative. Only `biji_normal` currently
supports explicit negative defect flags. This is necessary because the source
annotations provide one category per object, while SNI states that an object
can have multiple defects and uses the largest applicable defect value.

## This is not SNIB2 repeated

The failed classification experiment SNIB2 used a four-way router and
group-conditional classifiers that still reconstructed a 21-class leaf
distribution. It did not train explicit, observation-masked attributes.
SNIB2 lost 0.32 Macro-F1 and 12.37 Worst-F1 points against SNIB1 at seed 42;
that negative result remains valid.

This document does not revive that router. A future detector experiment, if
authorized, would retain one end-to-end YOLO detection graph and compare:

- the existing flat 21-class D0 head;
- a shared detection head with masked semantic attribute outputs and a
  deterministic SNI decoder.

The comparison must reuse the same backbone, neck, localization path, training
data, seed, and budget. It must report leaf SNI AP, attribute metrics,
lower-tail AP, operational defect-score error, parameters, and latency.

## Observability gate

Before model implementation:

- an SNI domain expert must approve every mapping;
- `kulit_tanduk` completeness labels must be reviewed against the fraction of
  an intact parchment, rather than apparent box size;
- physical foreign-matter size requires a scale reference or calibrated camera;
- local-cue attributes require a minimum visible-pixel rule;
- the deterministic decoder and SNI maximum-defect-value rule must be frozen.

Until those conditions are met, the YAML is a testable ontology specification,
not authorization for another long training run.
