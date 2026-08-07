from pathlib import Path

import torch
from PIL import Image

from coffee_detector.dataset import Box
from coffee_detector.dc2_crop import (
    CropRecord,
    RawObjectCropDataset,
    box_to_xyxy,
    build_local_classifier,
    classification_summary,
)
from coffee_detector.experiments.run_faruq_v3_dc2_crop_screening import RESOLUTIONS


def test_dc2_resolution_set_matches_predeclared_paper_screen() -> None:
    assert RESOLUTIONS == (32, 64, 128, 224)


def test_box_to_xyxy_uses_raw_bbox_and_clips_canvas() -> None:
    box = Box(class_id=0, x_center=0.5, y_center=0.5, width=0.4, height=0.2)
    assert box_to_xyxy(box, 100, 80, 1.0) == (30, 32, 70, 48)
    edge = Box(class_id=0, x_center=0.05, y_center=0.05, width=0.2, height=0.2)
    left, top, right, bottom = box_to_xyxy(edge, 100, 80, 1.0)
    assert left == 0 and top == 0
    assert right > left and bottom > top


def test_raw_crop_dataset_crops_before_resize(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (100, 80), color=(0, 0, 0))
    for x in range(30, 70):
        for y in range(32, 48):
            image.putpixel((x, y), (255, 0, 0))
    image.save(image_path)
    record = CropRecord(
        image_path=image_path,
        class_id=3,
        box=Box(class_id=3, x_center=0.5, y_center=0.5, width=0.4, height=0.2),
    )
    dataset = RawObjectCropDataset([record], 32, training=False, context=1.0)
    tensor, label = dataset[0]
    assert tensor.shape == (3, 32, 32)
    assert label == 3
    # A pure red crop after ImageNet normalization has positive red mean and
    # negative green/blue means; this checks crop-before-resize semantics.
    channel_means = tensor.mean(dim=(1, 2))
    assert channel_means[0] > 1.0
    assert channel_means[1] < 0.0
    assert channel_means[2] < 0.0


def test_local_classifier_output_contract_without_weight_download() -> None:
    model = build_local_classifier(21, imagenet_pretrained=False).eval()
    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 21)


def test_classification_summary_reports_macro_and_tail() -> None:
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )
    labels = torch.tensor([0, 1, 2, 2])
    summary = classification_summary(logits, labels, 3)
    assert 0.0 <= summary["worst_f1"] <= summary["macro_f1"] <= 1.0
    assert len(summary["per_class_f1"]) == 3
