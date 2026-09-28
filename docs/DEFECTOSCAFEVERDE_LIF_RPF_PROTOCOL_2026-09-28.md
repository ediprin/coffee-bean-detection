# DefectosCafeVerde LIF-RPF Seed-42 Protocol

## Question

Can a conservative luminance/illumination preprocessing cue improve the
paper-backed, physical-bean-grouped DefectosCafeVerde detector without
discarding native RGB evidence or modifying localization?

## Frozen candidate

`LIFRPF1` means **Luminance–Illumination Filtering with Raw-Preserving
Fusion**. A parameter-free operator estimates a large-scale illumination map
from Rec.709 luminance, flattens only the locally uneven component, and bounds
the luminance change to 0.08 in normalized image space. The identical scalar
change is applied to R, G, and B; channels are not filtered independently.

The detector always receives the original RGB image. Two maps derived from
the preprocessing operation—bounded luminance change and normalized log
reflectance—condition only the three classification branches. Native box/DFL
branches consume untouched RGB features. Each classification adapter uses a
bounded scalar gate initialized at zero, making the complete initial detector
output bitwise identical to `D0DIRECT`.

This is not AF2, wavelet preprocessing, chromatic conditioning, Retinexformer,
or a sampler. It has no auxiliary loss and no validation-selected threshold.

## Matched comparison

The existing `D0DIRECT` result from
`defectoscafeverde-grouped-dcwcf-direct-seed42-v1` is reused only after its
dataset-audit SHA, official-pretrained SHA, native-config SHA, training
schedule, and seed match the candidate contract exactly.

| Item | D0DIRECT | LIFRPF1 |
|---|---|---|
| Dataset | grouped DefectosCafeVerde development train/val | same |
| Initialization | official `yolo26n.pt` | same native detector tensors |
| Epochs | 50 | 50 |
| Image size | 640 | 640 |
| Seed | 42 | 42 |
| Localization | native RGB | native RGB |
| Classification | native RGB | native RGB + bounded LIF cue |

Test is not extracted or accessed.

## Static gates

Training is prohibited unless the audit proves:

- official pretrained SHA and 12-class ontology;
- identical model YAML, schedule, backbone tensors, and native head tensors;
- initial boxes and scores bitwise equal;
- active cue changes scores but not boxes;
- preprocessing is finite, active, and bounded;
- cue shape is exactly two channels;
- fusion gates start at zero and receive finite nonzero gradients;
- test remains inaccessible.

## Seed-42 decision

Primary metrics are Macro, Bottom-3, and Worst-class mAP50–95. Promotion uses
the frozen Defectos screen routes:

- overall route: Macro gain at least 0.5 point, Bottom-3 non-lower, and Worst
  drop no more than 1 point; or
- lower-tail route: Macro drop no more than 0.2 point, Bottom-3 gain at least
  1 point, and Worst gain at least 1 point.

Only a passing seed-42 result authorizes later paired confirmation. A failure
stops LIF-RPF. No locked test is opened in either case.
