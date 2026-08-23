# Faruq-v3 AF2_ORIENT + CMC0 Fusion Screening Protocol

Date: 2026-08-23
Status: **FROZEN**

## Purpose

This experiment tests the only successful variations from previous failed ablation branches:
1. `AF2_ORIENT`: Unsigned 180-degree phase folding (from `agent/af2-isolated-radial-orientation`), proven to slightly exceed base AF2 Macro mAP.
2. `CMC0`: Non-spatial classification channel-mixing capacity (from `agent/stb-capacity-causal-control`), proven to heavily lift Bottom-3 metrics at seed 42.

Since previous complex structural fusions (like STB spatial attention and Domain Generalization) failed to stack with AF2 due to gradient conflicts, this experiment isolates purely the *orientation feature space* and *classification head capacity*, evaluating if they can be summed without negative interaction.

## Configurations

- Base model: YOLO26n
- Input preprocessing: AF2_ORIENT (period=180, angular_bins=180)
- Detect Head: CMC0 (Zero-gated sequential linear channel mixing)

## Screening Gate (Seed 42)

The model will be evaluated strictly on the Faruq-v3 grouped validation split (seed 42).

**Acceptance Criteria:**
1. Macro mAP50-95 must exceed `D0FT` (86.69%) by at least +1.50 points.
2. Bottom-3 mAP50-95 must not be lower than `AF2FT` control.
3. No single class may drop more than 1 point from its baseline performance.

If successful, this will authorize the 3-seed paired confirmation.
