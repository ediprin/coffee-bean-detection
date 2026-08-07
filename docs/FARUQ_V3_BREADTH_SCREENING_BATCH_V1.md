# Faruq-v3 Breadth Screening Batch v1

## Objective

Run the already-implemented fine-grained detector candidates under one frozen seed-42 discovery protocol before any new combination model is designed.

The purpose is breadth, not final confirmation. The locked test split remains unavailable throughout this phase.

## Primary control

All common detector candidates are re-scored centrally against **D0FT**. ACMC1 is retained as context but is not the discovery control.

The central controller ignores historical branch-local `RETAIN/REJECT` wording and recomputes the same canonical gate for every arm:

- macro mAP50-95 delta versus D0FT >= -1.0 pp;
- bottom-3 delta >= -2.0 pp;
- worst-class delta >= -2.0 pp;
- at least one discovery signal: macro >= +0.20 pp, bottom-3 >= +0.50 pp, or worst >= +0.50 pp.

An arm is `RETAIN` only when all retention conditions and the discovery-signal condition pass.

## Frozen-code rule

`configs/breadth_screening/faruq_v3_batch_v1.json` stores an exact commit SHA for each candidate. The controller checks out that SHA directly rather than the moving branch tip. Therefore later edits to an experimental branch cannot silently change the frozen batch.

The controller itself never merges candidate branches. It checks out one candidate SHA, runs that branch's own screening runner, stores the result to Drive, then moves to the next candidate.

## Active common detector families

The frozen manifest contains 22 enabled common detector candidates:

1. SAFPN classification alignment
2. APCL
3. PCLDet prototype baseline
4. DRNet dual refinement/CML family
5. DRNet interaction verification
6. DSRDet FBNR
7. DSRDet SSCB/MSDA
8. leaf-preserving semantic auxiliary
9. semantic-guided leaf classifier (SG1)
10. LFDet CGFI
11. LFDet AFAB
12. LFDet FTIF
13. SFRNet S-Former
14. SFRNet C-Former
15. SFRNet SC-Former
16. SFRNet MRL
17. STB
18. IGEM
19. BHCL
20. Expert-style hierarchical visual prior
21. FSCE CPE
22. DCAL PWCA

A family runner may emit multiple arms; every emitted arm receives its own centralized decision row.

## Explicitly outside the common gate

- **GDSC1**: recorded but disabled because it is a YOLO26 transfer hypothesis, not literal anchor-geometry GDS.
- **DC2A-D**: recorded as diagnostics with distinct raw-crop/integrated protocols; they should not be mixed into the same detector-retention gate.
- **ACMC2/HCR**: already screened and failed earlier frozen criteria; not rerun.

## Chunking

The Colab notebook divides the 22 common candidates into three resumable chunks:

### Chunk 1 — semantics and metric geometry

`SG1, SSCB, MRL, APCL1, PCL1, CPE, BHCL, HIERVIP`

### Chunk 2 — representation and attention

`SAF1, DRNET, DRIV, SF1, CF1, SC1, STB1, IGEM, PWCA`

### Chunk 3 — regularization and frequency/language

`FBNR, SEMAUX, CG1, AFAB, FTIF`

Chunking is operational only. It does not change model selection or the gate. Every chunk writes into the same persistent batch root.

## Resume contract

The controller writes:

- `master_state.json`: per-candidate running/completed/failed ledger;
- `master_results.json`: standardized result rows plus controls and canonical gate;
- `master_results.csv`: flat table for comparison.

A candidate marked `completed` is skipped on rerun unless `--force-rerun` is supplied. Candidate-specific runners can separately resume from their own `last.pt`/`best.pt` behavior.

## Failure isolation

`--continue-on-error` allows a bad/legacy runner contract to be recorded as failed while the remaining candidates continue. A candidate failure is not interpreted as a scientific result. It must be fixed and rerun at the same frozen SHA or the batch manifest must be versioned explicitly.

## Test lock

The controller aborts if `<data-root>/test` exists. It passes `val`-oriented branch runners only and records `test_opened=false` and `test_images_accessed=false` in central outputs.

No candidate may be ranked, retained, tuned, or combined using locked-test information.

## After breadth screening

The next phase is not arbitrary module search. Use the central matrix to select roughly 5-8 mechanistically distinct survivors, then design 4-6 rational combinations (for example, a representation survivor plus a training-only regularizer) before paired multi-seed confirmation.
