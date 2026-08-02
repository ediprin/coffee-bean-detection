# Hong-to-YOLO26 Full Transfer And Conditional Ablation Protocol

Document ID: `HONG-YOLO26-TRANSFER`

Current version: `v1.2.0`

Status: frozen before training; static implementation gate passed, 2026-08-02.

Version history:

| Version | Date | Change |
|---|---|---|
| `v1.0.0` | 2026-08-02 | Initial component-first factorial protocol. |
| `v1.1.0` | 2026-08-02 | Replaced the expensive component-first sequence with a full-transfer-first fail-fast screen. Component and capacity controls are authorized only if the full Hong transfer passes. |
| `v1.2.0` | 2026-08-02 | Corrected the paper transfer after visual verification of Fig. 1 and Section 4.4: PConv modifies both box-regression and classification paths, and the frozen DSConv map follows the two backbone DSConv positions plus the first neck downsampling position shown in the paper. |

This protocol evaluates whether the mechanisms reported by Hong et al. transfer
to the repository's pinned YOLO26n detector. It is not a numerical reproduction
of Hong et al.'s seven-class YOLOv10 result.

Primary source:
[Hong et al. (2026)](<Hong et al. - 2026 - Automated detection of defective coffee beans based on improved YOLOv10 framework.pdf>).

## Question

When dataset, split, schedule, initialization, evaluation, and postprocessing
are fixed, do Hong-derived modules improve conditional fine-grained class
decisions on Faruq-v3 without degrading proposal accessibility, lower-tail AP,
or deployment efficiency?

## Why Direct Copying Is Invalid

Hong et al. and the pinned detector are structurally related but not identical.

| Component | Hong et al. | Pinned YOLO26n | Consequence |
|---|---|---|---|
| Assignment | YOLOv10 one-to-many plus one-to-one | YOLO26 end-to-end one-to-many plus one-to-one | Both branches must remain functional and separately audited |
| Classification branch | PConv is introduced in P3/P4/P5 detection heads | `Detect.cv3` and `one2one_cv3` already use depthwise plus pointwise blocks | Ordinary DWConv is not a Hong reproduction; PConv must be applied symmetrically to both class branches |
| Box branch | Hong describes PConv detection heads for classification and regression | `cv2` and `one2one_cv2` are independent box branches | A faithful full-head transfer must modify both box branches as well as both class branches |
| Bottleneck | SPPF is replaced by SPPF-Attention | YOLO26 already contains SPPF followed by C2PSA | The comparison is replacement versus the existing SPPF+C2PSA context path, not SPPF versus no context |
| DSConv | Variable Quantized Kernel plus Kernel/Channel Distribution Shifters; placed in early/middle backbone and neck downsampling | Standard full-precision convolution in downsampling layers; DWConv exists in the class head | Replacing layers with ordinary depthwise-separable convolution is an invalid proxy for the reported DSConv mechanism |
| Regression | Hong reports BCE, CIoU, and DFL in YOLOv10 | Pinned YAML uses `reg_max: 1`, making the DFL transform an identity | Hong loss weights must not be copied without a separate loss protocol |

Hong's paper gives feature-resolution roles for DSConv but does not provide an
official YOLO26 mapping. This study must therefore be called a **Hong-to-YOLO26
mechanism transfer**, not an exact architecture reproduction.

## Frozen Software And Foundation

- Repository branch: `agent/add-vadcp-pipeline`.
- Ultralytics: exactly `8.4.96`.
- Foundation YAML: `configs/coffee_fg/models/yolo26n-p3.yaml`.
- Initialization: official `yolo26n.pt` weights.
- Detection levels: P3, P4, and P5.
- End-to-end mode: enabled.
- One-to-many and one-to-one branches: both enabled during training.
- Reference model: Faruq-v3 `D0_seed42` checkpoint.
- Test split: unavailable and locked.

Do not use the failed CoffeeFG ROI wrapper as the implementation base. Hong
modules must be native feature/head modules so the inference graph remains a
single detector without proposal cropping.

## Dataset And Split

- Dataset: `faruq-development-v3-grouped`.
- Train: 1,665 parent-grouped images and 2,986 instances.
- Validation: 294 parent-grouped images and 526 instances.
- Classes: 21.
- Validation support: approximately 24--26 instances per class.
- Cross-split parent overlap: zero.
- Cross-split exact-hash overlap: zero.
- Test: absent from the development archive.

No augmentation siblings, generated crops, combined Adrian source, or dense
synthetic scenes may be introduced in this protocol.

## Architectural Arms

### Existing references

| Code | Model | Training status |
|---|---|---|
| `D0` | Native YOLO26n P3--P5 | Existing completed reference; do not retrain unless a paired software-control run becomes necessary |
| `D0-CA` | `D0` with frozen class-agnostic suppression at confidence 0.05 | Existing inference-only operational reference |

### Mechanism controls and Hong-derived arms

| Code | Change from native YOLO26n | Purpose |
|---|---|---|
| `HDW` | Ordinary full-precision depthwise-plus-pointwise replacements at the frozen DSConv transfer locations | Control whether any gain comes merely from depthwise factorization |
| `HDS` | Distribution-shift DSConv with VQK, KDS, and CDS at the same locations as `HDW` | Test Hong's distribution-shift mechanism |
| `HSC` | SPPF replacement with the same pooling path and a parameter-matched residual convolutional recalibration without spatial/channel gating | Capacity/depth control for SPPF-Attention |
| `HSA` | Hong-style SPPF-Attention at the same location and output width as `HSC` | Test spatial/channel recalibration rather than added depth |
| `HPC` | Standard-convolution detection-head blocks with the same input/output widths and block count as the PConv arm | Head mechanism control |
| `HP` | PConv detection-head blocks with partial ratio `r=1/4` in both regression and classification paths at P3/P4/P5 | Test the paper's full PConv detection-head mechanism |
| `HF` | Full transfer containing distribution-shift DSConv, SPPF-Attention, and symmetric PConv changes in both one-to-many and one-to-one detection heads | First and only initial training candidate; test whether the complete Hong architecture package transfers to YOLO26/SNI-21 |

`HDW`, `HSC`, and `HPC` are controls, not proposed final models.

## Required Transfer Map Before Training

Implementation must generate a machine-readable architecture audit containing
the following table. All cells must be resolved before any training command is
authorized.

| Item | Required record |
|---|---|
| DSConv source roles | Hong feature resolution and semantic role for every replaced layer |
| YOLO26 target layers | Exact YAML index, module path, stride, input channels, output channels, and parameter count |
| SPPF replacement | Exact YAML index, tensor shape, residual path, spatial gate, channel gate, and output width |
| PConv one-to-many | Exact `cv2` and `cv3` module paths for P3/P4/P5 |
| PConv one-to-one | Exact `one2one_cv2` and `one2one_cv3` module paths for P3/P4/P5 |
| Preserved prediction layers | State-dict keys and tensor hashes for the terminal 1x1 box/class prediction layers before training |
| Pretrained transfer | Loaded, missing, unexpected, and newly initialized keys by module |

The semantic starting map is:

- Hong backbone DSConv: the two downsampling operators following the initial
  stem and the first feature block (`model.1` and `model.3` in the pinned
  YOLO26n graph);
- Hong neck DSConv: the first bottom-up downsampling recovery operator
  (`model.17`); the later downsampling position is shown as SCDown in Hong's
  Fig. 1 and is therefore not relabeled DSConv;
- Hong SPPF-Attention: the single high-level SPPF bottleneck;
- Hong PConv: P3, P4, and P5 detection branches.

This semantic map is not permission to guess layer indices. The exact mapping
must be recorded from the pinned YAML and constructed model.

## Implementation Constraints

1. Keep the model end-to-end; no ROI crop, external classifier, or second-stage
   routing.
2. Preserve the number of detection levels and their strides.
3. Apply every class-head change to both one-to-many and one-to-one branches.
4. Do not change the box loss, assignment, input resolution, augmentation,
   optimizer, or postprocessing inside a module comparison.
5. `HP` applies the same PConv rule to box and class branches. The terminal 1x1
   prediction layers, class count, `reg_max`, and branch outputs must remain
   equal to `D0` before training.
6. `HSA` must match `HSC` in tensor widths and block placement.
7. `HDS` must use the paper's distribution-shift parameterization. A
   `DWConv+PWConv` implementation must be labeled `HDW`, never `HDS`.
8. Newly initialized weights and transferred pretrained weights must be listed
   explicitly.
9. Every model must export an artifact manifest and resolved configuration.
10. Checkpoints must be written to the shared project Drive and support resume;
    ephemeral `/content` is not an artifact authority.

## Static Gate: No GPU Training

Each arm must pass all of the following before training:

1. import and YAML/model construction;
2. forward pass at batch 1 and batch 2;
3. backward pass with finite gradients;
4. identical output schema for one-to-many and one-to-one branches;
5. expected P3/P4/P5 tensor shapes;
6. state-dict save and reload equivalence;
7. checkpoint-resume smoke test;
8. pretrained-load audit;
9. parameter count and FP32 model size;
10. same-device batch-1 latency smoke benchmark;
11. terminal box/class prediction-layer preservation check for `HPC` and `HP`; and
12. tests proving that an arm cannot silently fall back to the native module.

Any failure stops that arm. Training is not a debugging mechanism for an
unverified architecture.

## Training Schedule

The one-seed screen uses the same completed-baseline schedule:

- seed: 42;
- epochs: 50;
- patience: 15;
- image size: 640;
- batch: 16;
- workers: 2;
- optimizer: Ultralytics `auto`;
- deterministic mode: enabled;
- close mosaic: final 10 epochs;
- pretrained initialization: enabled;
- evaluation split: validation only.

The fail-fast sequence is:

1. reuse the completed `D0_seed42` baseline without retraining;
2. train only `HF_seed42`, containing all three faithfully transferred Hong
   mechanisms;
3. stop the Hong study immediately if `HF_seed42` fails any one-seed gate;
4. if `HF_seed42` passes, run conditional removal/mechanism controls to identify
   whether DSConv, SPPF-Attention, and PConv contribute beyond ordinary
   factorization, capacity, or head depth; and
5. expand only the mechanism-qualified full candidate to seeds 123 and 2026.

`HDW`, `HDS`, `HSC`, `HSA`, `HPC`, and `HP` are therefore deferred diagnostic
arms, not prerequisites for the initial `HF` screen. This ordering minimizes
GPU cost while still implementing the complete nearest prior art before a
thesis-specific modification is proposed.

If a control itself improves the baseline but the corresponding Hong mechanism
does not improve the control, the conclusion is added capacity/factorization,
not support for the Hong mechanism.

## Evaluation

### Primary detection metrics

- mAP50-95;
- mAP50;
- precision and recall;
- macro per-class AP50-95;
- bottom-three class AP50-95;
- worst-class AP50-95 and class name;
- complete per-class AP table.

### Conditional diagnostic metrics

At candidate count 500 and the frozen diagnostic matching rules, report:

- proposal accessibility;
- conditional top-1 class accuracy;
- classification-error headroom;
- proposal miss rate;
- localized wrong-class rate; and
- class-pair confusion for the five weakest classes.

### Operational metrics

Use the already selected development operating point, confidence 0.05 with
class-agnostic suppression, without retuning it per model. Report:

- correct-decision precision, recall, and F1;
- predictions per image;
- duplicate/competing-class suppression counts; and
- proposal accessibility after suppression.

Curves may be reported as threshold-free diagnostics. A new operating threshold
must not be selected independently for each candidate during the architecture
screen.

### Efficiency metrics

- total and trainable parameters;
- FP32 state-dict size;
- FLOPs using one frozen profiler;
- batch-1 latency after warm-up;
- batch-32 throughput when memory permits; and
- peak CUDA memory.

All latency comparisons must use the same T4 runtime, PyTorch/Ultralytics
version, image size, precision mode, warm-up, and repetition count.

## One-Seed Gates

A Hong-derived arm passes `D0` only if all conditions hold:

1. macro AP50-95 improves by at least 0.50 percentage points;
2. conditional top-1 class accuracy improves by at least 2.00 points;
3. bottom-three AP50-95 does not decrease by more than 1.00 point;
4. worst-class AP50-95 does not decrease by more than 2.00 points;
5. proposal accessibility does not decrease by more than 1.00 point;
6. operational correct-decision F1 does not decrease; and
7. batch-1 latency does not increase by more than 25%, unless the arm is
   explicitly retained as a non-edge accuracy candidate.

It passes its mechanism control only if:

1. macro AP50-95 is higher than the paired control;
2. conditional top-1 class accuracy is higher than the paired control; and
3. bottom-three and worst-class preservation gates still hold.

No single favorable headline metric overrides a failed lower-tail or
conditional-classification gate.

## Multi-Seed Confirmation

Only the best one-seed mechanism-qualified arm may expand to seeds 123 and
2026. Confirmation requires:

- mean macro AP50-95 gain of at least 0.50 points and improvement in at least
  two of three seeds;
- mean conditional top-1 gain of at least 2.00 points and improvement in at
  least two of three seeds;
- mean bottom-three degradation no worse than 1.00 point;
- mean worst-class degradation no worse than 2.00 points;
- mean proposal accessibility degradation no worse than 1.00 point; and
- no seed with a catastrophic lower-tail decrease greater than 10 points.

Report paired mean, standard deviation, per-seed deltas, and class-level
changes. One strong seed cannot compensate for two failed seeds.

## Test Lock

Test remains locked throughout static audit, one-seed screening, mechanism
controls, and multi-seed confirmation.

Test may be opened once only after:

1. one arm passes all multi-seed gates;
2. its configuration and checkpoint-selection rule are frozen;
3. its efficiency comparison is complete;
4. no further architecture or threshold tuning is planned; and
5. an explicit test-unlock decision is recorded in a result document.

If no arm passes, the correct outcome is a preserved negative result. Do not
substitute a new dataset, merge sources, or open test to rescue the method.

## Interpretation Rules

- `HDS > HDW > D0`: evidence for distribution shift beyond ordinary depthwise
  factorization.
- `HDW > D0`, but `HDS <= HDW`: evidence for factorization, not Hong DSConv.
- `HSA > HSC > D0`: evidence for attention beyond added bottleneck capacity.
- `HSC > D0`, but `HSA <= HSC`: evidence for depth/capacity, not attention.
- `HP > HPC > D0`: evidence for partial-channel processing beyond head depth.
- `HPC > D0`, but `HP <= HPC`: evidence for head capacity, not PConv.
- higher mAP with unchanged conditional top-1: likely localization/ranking gain,
  not resolution of fine-grained classification.
- higher conditional top-1 with lower proposal accessibility: an unacceptable
  trade unless a separately declared operating objective permits it.
- `HF > D0` at one seed: evidence that the complete Hong-derived package is
  worth decomposing; it is not yet evidence that every included component is
  useful.
- `HF` passing its later removal/mechanism controls: evidence for a transferable
  package with identified contributing mechanisms.

## Stopping Rules

Stop the Hong transfer study when any of the following occurs:

1. `HF_seed42` fails any one-seed baseline gate;
2. `HF` passes initially but fails the required removal/mechanism controls;
3. the qualified full candidate fails multi-seed confirmation;
4. implementation cannot faithfully represent VQK/KDS/CDS and would only be an
   unlabeled DWConv proxy; or
5. artifact persistence or dataset provenance cannot be guaranteed.

After a stop, record the negative result. Do not proceed to further component
experiments or the locked test.

## Required Outputs

For every attempted arm preserve:

- resolved model and training configuration;
- architecture transfer map;
- pretrained-load audit;
- parameter/FLOP/latency report;
- `last.pt`, `best.pt`, history, and artifact manifest outside Git;
- validation summary and complete per-class metrics;
- conditional diagnostic report;
- operational report at the frozen operating point; and
- a PASS/FAIL decision JSON containing every criterion.

The verified result must be documented in Git before another candidate is
authorized.

## Recorded outcome — 2026-08-02

`HF_seed42` failed the frozen validation gate. Proposal accessibility improved
by 11.79 points, but Macro mAP50-95 fell 4.92 points, conditional top-1 fell
16.89 points, bottom-3 mAP50-95 fell 23.58 points, worst-class mAP50-95 fell
33.01 points, operational correct-decision F1 fell 11.05 points, and latency
rose to 1.4287x the D0 baseline. Test access remained false. Per the stopping
rule, the Hong transfer study stops without additional seeds or component
controls. The full record is in
`docs/HONG_YOLO26_TRANSFER_RESULT_2026-08-02.md`.
