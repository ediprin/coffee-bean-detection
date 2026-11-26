# Auliya Robusta SNI Frequency-Preprocessing Screen

Protocol `auliya-robusta-sni-frequency-v1`, frozen before model training on 2026-11-26.

Does fixed input preprocessing improve YOLO instance segmentation of visual
Robusta defects when validation uses a different acquisition group?

The source is [Auliya et al. on Zenodo](https://zenodo.org/records/24346584).
The matching [Roboflow v2 YOLO export](https://universe.roboflow.com/yudha-workspace/primer-robusta_sni_dataset-pwzyy/dataset/2)
contains 2,148 original-resolution images. The preparation command verifies
both archive hashes and the authors' class order. It retains the released
polygons and checks the instance count of each development image.

| Role | Acquisition group | Images | Content |
| --- | --- | ---: | --- |
| Train | B (V3 and V4) | 375 | Single beans and groups of objects of the same class |
| Validation | C (V5) | 420 | Touching objects with mixed classes |
| Held-out final evaluation | A (V1 and V2) | 1,351 | Single objects photographed under several rotations and viewing angles |

This keeps the development pool and held-out group from official fold A,
but splits development by acquisition group. Both development groups contain
all 21 classes. The authors report that they reused specimens for the three
soil, stone and twig classes (IDs 18–20). Retain all 21 labels for training
and report primary metrics over the 18 classes unaffected by that known reuse.
The 21-class aggregate is supplementary. The comparison concerns visual
object segmentation, with mask mAP50–95 and lower-tail AP as endpoints.

| Arm | Input to the same YOLO26n-seg network |
| --- | --- |
| RGB | Original normalized RGB tensor |
| AF2 | Historical patch-based Fourier AF2 with the existing settings |
| HAAR | Two-scale signed Haar luminance detail with fixed gains 0.25 and 0.125 |

All three operators run immediately before the first network layer, after the
shared Ultralytics spatial transforms and normalization. Haar adds the same
bounded luminance offset to each channel, which preserves channel differences.
All arms use official `yolo26n-seg.pt`, seed 42, 50 epochs, 640 pixels, batch 8,
and matched default augmentations under Ultralytics 8.4.96. Each arm starts
from the official pretrained weights. Report validation metrics from
`last.pt` at the fixed epoch budget.

The first comparison reports 18-class mask mAP50–95, Bottom-3, Worst, and AP
by class. Pay particular attention to single-hole and partly-black beans.
This is one development screen. Any further claim requires an independent
evaluation of the selected method and the RGB control on group A.

## Commands

Run from the repository root with the package installed.

```sh
python -m coffee_detector.experiments.prepare_auliya_robusta_frequency \
  --metadata-zip splits_metadata_results_scripts.zip \
  --images-zip primer-robusta_sni_dataset.v2i.yolov8.zip \
  --output auliya-development

python -m coffee_detector.experiments.run_auliya_robusta_frequency \
  --arm RGB --development auliya-development --weights yolo26n-seg.pt \
  --output runs/auliya-frequency --device 0
```

Use the same inputs and output root for `--arm AF2` and `--arm HAAR`.
Each arm writes its checkpoint and `validation_result.json` in its own run
directory.

## Google Colab

Upload the verified `auliya-robusta-sni-frequency-colab-v1.zip` once to
`Coffee_Bean_Detection/bundles/` in the shared project. The bundle contains
the development data, the matching runner source, and the official initial
weights. Its SHA-256 is
`b8795e189b85a13201e54a562cbe0f42fa7e0f2bfc0eb4b6d04f85d2b4cf57be`.

Open one GPU notebook per arm:

- `notebooks/Auliya_Robusta_SNI_RGB_Seed42_Colab.ipynb`
- `notebooks/Auliya_Robusta_SNI_AF2_Seed42_Colab.ipynb`
- `notebooks/Auliya_Robusta_SNI_HAAR_Seed42_Colab.ipynb`

The arms can run concurrently in separate GPU sessions. Use one active
session per arm. Each arm writes its log, progress, CSV, and optimizer-bearing
`weights/last.pt` to its own `experiments/auliya-robusta-sni-frequency-v1/`
subdirectory in the shared project. After a session ends, rerun the same
notebook against the same output folder. It continues from the last
completed checkpoint or reuses the completed validation report. A resumed
run restores model and optimizer state at an epoch boundary. The training
log records the interruption and resume point.
