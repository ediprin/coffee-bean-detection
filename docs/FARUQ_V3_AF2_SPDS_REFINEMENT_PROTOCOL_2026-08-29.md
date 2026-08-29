# Faruq-v3 AF2-SPDS Refinement Protocol

Status: frozen after the original AF2-SPDS seed-42 failure and before refinement training.  
Test: locked.

## Evidence motivating the refinement

Against the matched `AF2BASE`, the original `AF2SPDS` changed Macro/Bottom-3/Worst
by -0.24/+3.83/+5.91 points. Against generic `AF2RGBDS`, it changed the same
metrics by +0.01/+1.05/+0.82 points. Thus AF2-specific supervision has a real
lower-tail signal, but the implementation sacrifices some average performance.

Code audit identified two separable defects:

1. `AF2(x)-x` equals raw RGB multiplied by the normalized recovered gate; it is
   not an isolated frequency target and remains strongly coupled to RGB.
2. Auxiliary gain 0.10 remains active through the final update, so the
   reconstruction objective can keep pulling features away from the native
   detection optimum after it has already shaped the representation.

## Factorized arms

| Arm | Only change from original AF2SPDS |
|---|---|
| `AF2CUE1` | target is the pure normalized AF2 recovery gate |
| `AF2DECAY1` | original target retained; gain decays cosine from 0.10 to zero over the final 10 epochs |

Both begin from the same original AF2 seed-42 checkpoint and use the same
30-epoch continuation schedule, decoder capacity, seed, and development split.
They can run in parallel. No combination is allowed in this screen.

## Kill gate

A refinement is retained only if:

- Macro is within 0.1 point of matched `AF2BASE`;
- Bottom-3 is within 0.5 point of original `AF2SPDS`;
- Worst is within 0.5 point of original `AF2SPDS`; and
- at least one headline metric improves over original `AF2SPDS`.

Only a retained winner may receive a future paired confirmation protocol. No
test access or further tuning is authorized by this screen.
