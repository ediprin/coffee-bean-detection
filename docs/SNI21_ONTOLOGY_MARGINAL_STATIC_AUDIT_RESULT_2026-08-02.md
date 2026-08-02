# SNI-21 Ontology-Marginal Static Audit Result

Date: 2026-08-02

Protocol: `sni21-ontology-marginal-static-audit-v1`

Decision: **PASS**

Training executed: no

Dataset accessed: no

Test images accessed: no

## Verified gates

All static gates passed:

- identical model YAML and pretrained initialization;
- identical 50-epoch training schedule;
- identical parameter count: `2,511,990` for D0, C0, and S0;
- identical state-dict schema;
- bit-identical inference output after loading the same weights;
- identical task masks, task weights, and auxiliary gain for C0 and S0;
- C0 uses identity control and S0 uses semantic marginalization.

Maximum inference difference against D0 was exactly `0.0` for C0 and S0 on
the frozen synthetic-tensor equivalence check. The method changes training loss
only; it does not add an inference branch or parameter.

## Consequence

The implementation is structurally eligible for a validation-only seed-42
screening runner. This static result does not itself authorize training. The
runner must preserve the frozen Faruq-v3 split, D0 checkpoint provenance,
metrics, acceptance gates, and test lock defined by
`SNI21_ONTOLOGY_MARGINAL_PROTOCOL.md`.
