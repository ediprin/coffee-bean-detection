# DefectosCafeVerde Grouped — AF2 Direct Paired Protocol

Frozen: 2026-09-18

## Question

Does the retained parameter-free AF2 frontend improve fine-grained detection
on the paper-backed DefectosCafeVerde dataset when physical-bean identity,
initial weights, target-head randomness, schedule, and validation endpoint are
controlled?

## Arms

| Arm | Detector | Input |
|---|---|---|
| `D0DIRECT` | YOLO26n P3–P5 | native RGB |
| `AF2DIRECT` | identical YOLO26n P3–P5 | retained AF2 frontend |

Both arms start directly from the same official `yolo26n.pt`. The 12-class
target head is initialized inside the same isolated RNG state. AF2 has no
learned parameters; no Faruq, coffee-standard, D0FT, or historical AF2
checkpoint is used.

## Dataset and lock

- Dataset: `defectoscafeverde-grouped-physical-v1`
- Development data: 2,827 train and 808 validation images
- Test: 403 images, absent from the runtime development root
- Inferred physical groups do not cross splits
- Seed-42 is screening only
- Maximum 50 epochs, image size 640, batch 16, patience 15, optimizer `auto`
- Model selection uses validation only

## Metrics

Primary: Macro mAP50–95. Lower-tail safeguards: Bottom-3 class mAP50–95 and
Worst-class mAP50–95. All 12 classes must be present at validation.

## Frozen promotion gate

Promotion to paired seeds 123 and 2026 occurs through either route:

1. Overall route: Macro gain at least +0.5 point, Bottom-3 not lower, and
   Worst-class drop no greater than 1 point.
2. Lower-tail route: Macro drop no greater than 0.2 point, Bottom-3 gain at
   least +1 point, and Worst-class gain at least +1 point.

A seed-42 pass is not a superiority claim. Only a later paired three-seed pass
may authorize one final test evaluation. No threshold or schedule may be
changed after observing seed-42 validation.
