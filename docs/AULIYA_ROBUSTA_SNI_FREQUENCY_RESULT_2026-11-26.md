# Auliya Robusta SNI frequency preprocessing — seed 42

All three training arms completed the fixed 50-epoch budget. AF2 improved
mean validation mask AP but reduced both lower-tail metrics. Haar improved
both lower-tail metrics but reduced mean AP. Neither variant dominates RGB.
These are development results; no test images were accessed.

## Paired validation screen

The [frozen protocol](AULIYA_ROBUSTA_SNI_FREQUENCY_PROTOCOL_2026-11-26.md)
trained YOLO26n-seg from the same official pretrained checkpoint on group B
(375 images) and evaluated `last.pt` on group C (420 images, 1,265 annotated
instances). The run reports agree on the development bundle and contract,
initial weights, seed 42, 50 epochs, 640 resolution, batch 8, and Ultralytics
8.4.96. Training and original validation used Tesla T4 runtimes.

| Arm | Mask mAP50–95, 18 classes | Bottom-3 AP | Worst AP | Mask mAP50–95, all 21 |
| --- | ---: | ---: | ---: | ---: |
| RGB | 21.2743% | 6.4715% | 5.3023% | 22.1578% |
| AF2 | 21.9779% | 5.5795% | 4.5660% | 23.0088% |
| HAAR | 20.7022% | 7.3918% | 5.6903% | 22.2683% |

Relative to RGB, AF2 changed the three primary metrics by **+0.70, −0.89,
and −0.74 percentage points**; Haar changed them by **−0.57, +0.92, and
+0.39 points**. The 21-class result is supplementary because the authors
document specimen reuse across acquisition groups for the three soil, stone,
and twig classes.

## Validation-only error audit

The local CPU rerun verified each checkpoint against the reported SHA-256
and used the same Ultralytics version, 640-pixel input, and all 420 group-C
images. The 18-class mask mAP reproduced the original endpoint within
0.0004 percentage points per arm. No weights changed and no images from
held-out group A were read.

For each annotated object, the audit checked the returned candidates at two
diagnostic score thresholds (0.001 and 0.25). A usable candidate has the
correct class and both box and mask IoU ≥0.50. These are per-object candidate
availability counts, not one-to-one AP matches. The thresholds describe score
behavior, not proposed deployment settings.

### Biji Hitam Sebagian

This class has 59 train and 60 validation annotations. The four columns
under score ≥0.25 partition the 60 validation objects in each arm.

| Arm | Usable ≥0.001 | Usable ≥0.25 | Only wrong-class box ≥0.25 | Correct-class box, weak mask ≥0.25 | No overlapping box ≥0.25 | Box AP50 = mask AP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB | 59 | 2 | 23 | 0 | 35 | 6.33% |
| AF2 | 57 | 2 | 24 | 0 | 34 | 5.97% |
| HAAR | 58 | 1 | 25 | 0 | 34 | 6.82% |

Usable candidates exist at low scores for most targets. At 0.25 very few
survive, and wrong-class boxes are common. In the AF2 wrong-class-only
bucket, the highest-IoU overlapping box was Biji Hitam Pecah in 10 cases.
This points to confidence and class confusion, not an absent mask at IoU
0.50, as the immediate failure mode. It does not establish whether appearance
overlap, annotation, or learned scores explain the confusion.

### Biji Hitam Pecah

For this class (60 train and 60 validation annotations), usable candidates
at score ≥0.25 rose from 10/60 for RGB to 35/60 for AF2 and 17/60 for HAAR.
Mask AP50–95 was 14.73%, 33.84%, and 25.03%, respectively. The Fourier gain in
this class did not transfer to Biji Hitam Sebagian.

## Provenance and limits

[Zenodo v2](https://zenodo.org/records/24346584) documents the source
collection and acquisition groups; the matching Roboflow v2 export supplies
the image and YOLO polygon files. A provenance comparison found near-identical
visual matches in Faruq-v3 development for 722 of 795 Auliya B/C images,
including 314 of 420 group-C validation images matched in Faruq-v3 training.
The comparison does not establish annotation equivalence or independence for
group A against historical runs. These three models did not initialize from
Faruq weights, but the study is not an independent external replication of
prior Faruq experiments.

This single-seed screen does not select an overall winner. A useful next
diagnostic is to review high-scoring false positives and confused validation
images before changing preprocessing or running additional seeds.

Raw logs, metrics, and checkpoint artifacts remain in the shared project at
`experiments/auliya-robusta-sni-frequency-v1/{RGB,AF2,HAAR}/`. Each arm has
`validation_result.json`, `run_contract.json`, and `weights/last.pt`. No dataset
or model weights are included in Git.
