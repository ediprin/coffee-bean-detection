# Auliya official fold A — paired frequency preprocessing screen

Frozen before training. This adapts `scripts/training/run_baseline_v4.py`
from [Auliya et al., Zenodo v2](https://zenodo.org/records/24346584), which is
the official 21-class runner. `run_jalur2.py` is the supplementary runner.
The previous [B→C pilot](AULIYA_ROBUSTA_SNI_FREQUENCY_RESULT_2026-11-26.md)
is not a reproduction of the authors' benchmark and provides no parent
checkpoint for this screen.

## Paired experiment

- Official fold A: exactly 676 train and 119 validation images from groups B/C,
  using the authors' file lists. The 1,351 held-out group-A images are never
  extracted or used. The original Roboflow folder called `test` may contain B/C
  images; fold membership, not that source directory name, controls access.
- Same YOLO26s-seg official weights for all arms, seed 0 (one of the authors'
  seeds), Ultralytics 8.4.210, 640 px, batch 16, four workers, maximum 80
  epochs, patience 25, AdamW, and the authors' other explicit hyperparameters
  verbatim. Run on a matched GPU and PyTorch version across arms. The
  corresponding Colab runtime is recorded and need not match the authors'
  documented PyTorch 2.5.1, so this is a controlled adaptation, not a claim
  of exact reproduction of the published test scores.
- `RGB` is the native segmentation model; `AF2` and `HAAR` enable the same
  parameter-free input filters used in the pilot. Initialization and native
  weight tensors are checked across all three. No arm uses a coffee checkpoint.
- Keep the author's selection rule (`best.pt`). The runner deliberately
  replaces the automatic test call with validation-only reporting. Metrics:
  18-class mask mAP50–95, Bottom-3, Worst, all-21 aggregate and classwise AP.
  Check Biji Hitam Sebagian and Biji Hitam Pecah in particular. Validation
  also selects checkpoints, so do not treat it as independent confirmation.
- Report gains and regressions against RGB. A Macro gain with a tail loss
  is a trade-off, not a general improvement. Do not auto-promote, expand
  seeds, tune on group A, or open it based on this single-seed screen.

## Colab and runtime guardrails

Open `notebooks/Auliya_Official_Fold_A_Frequency_Colab.ipynb` in a fresh
Colab runtime. The small source ZIP belongs beside the previously uploaded
`auliya-robusta-sni-frequency-colab-v1.zip` in `Coffee_Bean_Detection/bundles/`.
The notebook verifies the source ZIP, then downloads and verifies the public
metadata and official YOLO26s-seg pretrained asset. The data prep reconstructs
the official fold from the verified pilot bundle without downloading a new image archive.

Preflight is the default and does not train. Explicitly enable training only
after preflight passes. Run one session per arm, and keep the same GPU type
and software versions. Each arm publishes optimizer-bearing `last.pt`,
`best.pt`, a checkpoint-bound progress marker, and `results.csv` under
`experiments/auliya-official-fold-a-frequency-v1/<ARM>/`. Interruption between
complete epochs can be resumed with the same contract; an incomplete or
mismatched checkpoint publication stops instead of silently restarting.
No test runner is provided by this notebook.

For verification outside Colab, run the two modules (make `src/` importable
and use Ultralytics 8.4.210 for model preflight):

```sh
python -m coffee_detector.experiments.prepare_auliya_official_fold \
  --metadata auliya-zenodo-24346584-metadata.zip \
  --pilot-bundle auliya-robusta-sni-frequency-colab-v1.zip --output fold_a
python -m coffee_detector.experiments.run_auliya_official_frequency \
  --development fold_a --weights yolo26s-seg.pt
```
