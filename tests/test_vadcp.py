from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from coffee_detector.audit_vadcp import audit_vadcp_dataset, decode_uncompressed_rle
from coffee_detector.audit_vadcp_realism import (
    _comparison,
    audit_vadcp_realism,
    build_realism_reference,
)
from coffee_detector.generate_vadcp_dataset import (
    _calibrated_canvas_size,
    generate_vadcp_dataset,
)
from coffee_detector.evaluate_visibility import (
    count_metrics,
    evaluate_predictions_by_visibility,
)
from coffee_detector.prepare_object_library import prepare_classification_library
from coffee_detector.run_vadcp_visual_audit import run_vadcp_visual_audit
from coffee_detector.run_cutout_visual_audit import run_cutout_visual_audit
from coffee_detector.vadcp.compositor import (
    CompositionSpec,
    _choose_unique_cutouts,
    _rotation_and_projected_size,
    _transform_cutout,
    compose_scene,
)
from coffee_detector.vadcp import library as library_module
from coffee_detector.vadcp.library import (
    backfill_intrinsic_geometry,
    load_object_library,
    prepare_yolo_library,
)
from coffee_detector.vadcp.masks import (
    binary_mask_rle,
    crop_to_mask,
    estimate_foreground_mask,
    largest_component,
    mask_bbox,
)
from coffee_detector.vadcp.profile import (
    SceneCalibration,
    build_scene_calibration,
    load_scene_calibration,
    save_scene_calibration,
)
from coffee_detector.vadcp.types import Cutout
from coffee_detector.dataset import discover_layout


def _write_classification_assets(root: Path) -> None:
    colors = {"normal": (80, 130, 60, 255), "defect": (90, 55, 35, 255)}
    for split in ("train", "test"):
        for class_name, color in colors.items():
            folder = root / split / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                image = Image.new("RGBA", (48, 36), (0, 0, 0, 0))
                draw = ImageDraw.Draw(image)
                draw.ellipse((5 + index, 4, 42, 31), fill=color)
                image.save(folder / f"{class_name}-{index}.png")


def _write_real_dataset(root: Path) -> None:
    names = {0: "defect", 1: "normal"}
    (root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for split_index, split in enumerate(("train", "val", "test")):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        image = Image.new(
            "RGB", (128, 128), (220 - split_index * 12, 222, 224 + split_index * 4)
        )
        image.save(image_dir / f"{split}.jpg")
        (label_dir / f"{split}.txt").write_text(
            "0 0.30 0.50 0.20 0.25\n1 0.70 0.50 0.20 0.25\n",
            encoding="utf-8",
        )


def _write_yolo_cutout_source(root: Path, images_per_class: int = 8) -> None:
    names = {0: "defect", 1: "normal"}
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "train/images", "names": names},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    image_dir = root / "train" / "images"
    label_dir = root / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    colors = {0: (80, 45, 25), 1: (65, 125, 55)}
    for class_id in names:
        for index in range(images_per_class):
            image = Image.new("RGB", (128, 128), "white")
            draw = ImageDraw.Draw(image)
            draw.ellipse((32, 40, 96, 88), fill=colors[class_id])
            stem = f"class{class_id}-{index}"
            image.save(image_dir / f"{stem}.png")
            (label_dir / f"{stem}.txt").write_text(
                f"{class_id} 0.5 0.5 0.5 0.375\n", encoding="utf-8"
            )


def test_uncompressed_rle_round_trip() -> None:
    mask = np.zeros((7, 9), dtype=bool)
    mask[1:5, 2:7] = True
    mask[3, 4] = False

    decoded = decode_uncompressed_rle(binary_mask_rle(mask))

    assert np.array_equal(decoded, mask)


def test_rectangular_images_use_isotropic_pixel_geometry(tmp_path: Path) -> None:
    root = tmp_path / "rectangular"
    image_root = root / "train" / "images"
    label_root = root / "train" / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "names": {0: "bean"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (200, 100), "white").save(image_root / "scene.jpg")
    (label_root / "scene.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    layout = discover_layout(root)
    calibration = build_scene_calibration(layout, background_samples=1)
    realism = build_realism_reference(root)

    # The normalized YOLO box is 0.2 x 0.2, but in pixels it is 40 x 20.
    assert calibration.bbox_width_height_ratios == pytest.approx((2.0,))
    assert calibration.object_long_sides == pytest.approx((0.2,))
    assert calibration.canvas_width_height_ratios == pytest.approx((2.0,))
    assert _calibrated_canvas_size(640, calibration) == (640, 320)
    assert realism["canvas_absolute_aspect_ratio"] == pytest.approx([2.0])
    assert realism["absolute_aspect_ratio"] == pytest.approx([2.0])
    assert realism["long_side_fraction"] == pytest.approx([0.2])
    assert realism["bbox_area_fraction"] == pytest.approx([0.04])


def test_realism_comparison_uses_practical_equivalence_margin() -> None:
    real = np.linspace(0.0, 1.0, 2000).tolist()
    small_raster_shift = (np.linspace(0.0, 1.0, 2000) + 0.06).tolist()
    large_shift = (np.linspace(0.0, 1.0, 2000) + 0.50).tolist()

    practical = _comparison(real, small_raster_shift, random.Random(11))
    rejected = _comparison(real, large_shift, random.Random(11))

    assert practical["normalized_quantile_distance"] < 0.15
    assert practical["status"] == "within_practical_equivalence_margin"
    assert rejected["status"] == "shifted"


def test_component_selection_prefers_annotated_box_center() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[2:18, 2:12] = True
    mask[7:14, 22:28] = True

    largest = largest_component(mask)
    centered = largest_component(mask, preferred_point=(25.0, 10.0))

    assert largest[:, 2:12].sum() == 160
    assert centered[:, 22:28].sum() == 42
    assert centered[:, 2:12].sum() == 0


def test_cutout_alpha_is_feathered_inward_without_expanding_mask() -> None:
    image = Image.new("RGB", (32, 24), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((5, 4, 26, 19), fill=(75, 45, 25))
    mask = np.zeros((24, 32), dtype=bool)
    mask[4:20, 5:27] = True

    rgba, cropped_mask = crop_to_mask(image, mask, padding=2)
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)

    assert np.all(alpha[~cropped_mask] == 0)
    assert np.any((alpha[cropped_mask] > 0) & (alpha[cropped_mask] < 255))
    assert np.all(alpha[cropped_mask] > 0)


def test_foreground_mask_and_matte_do_not_retain_white_rim() -> None:
    image = Image.new("RGB", (40, 32), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 7, 31, 24), fill=(80, 45, 25))

    mask = estimate_foreground_mask(image, threshold=24)
    assert mask_bbox(mask) == (8, 7, 24, 18)
    rgba, _ = crop_to_mask(image, mask, padding=2)
    pixels = np.asarray(rgba, dtype=np.uint8)
    positive_alpha = pixels[:, :, 3] > 0

    assert int(pixels[:, :, :3][positive_alpha].max()) < 160


def test_calibrated_rotation_matches_signed_bbox_ratio() -> None:
    mask = np.zeros((64, 96), dtype=bool)
    yy, xx = np.ogrid[:64, :96]
    mask[((xx - 48) / 38) ** 2 + ((yy - 32) / 15) ** 2 <= 1] = True

    for target_ratio in (2.0, 0.5):
        _, projected = _rotation_and_projected_size(
            mask,
            random.Random(31),
            target_ratio,
        )
        assert projected[0] / projected[1] == pytest.approx(
            target_ratio, abs=0.04
        )


def test_transformed_cutout_matches_ratio_and_final_long_side(
    tmp_path: Path,
) -> None:
    # A diagonal source catches the y-down mask versus PIL rotation sign.
    # A horizontal ellipse cannot distinguish the two symmetric signs.
    mask = np.zeros((96, 96), dtype=bool)
    yy, xx = np.ogrid[:96, :96]
    angle = np.deg2rad(37.0)
    centered_x = xx - 48
    centered_y = yy - 48
    major_axis = centered_x * np.cos(angle) + centered_y * np.sin(angle)
    minor_axis = -centered_x * np.sin(angle) + centered_y * np.cos(angle)
    mask[(major_axis / 38) ** 2 + (minor_axis / 15) ** 2 <= 1] = True
    pixels = np.zeros((96, 96, 4), dtype=np.uint8)
    pixels[mask, :3] = (80, 45, 25)
    pixels[mask, 3] = 255
    asset_path = tmp_path / "elongated.png"
    Image.fromarray(pixels, mode="RGBA").save(asset_path)
    cutout = Cutout("asset", 0, "bean", asset_path, "source")
    spec = CompositionSpec(
        canvas_size=(128, 128),
        object_range=(1, 1),
        use_shadows=False,
    )

    for target_ratio in (2.0, 0.5):
        calibration = SceneCalibration(
            scene_counts=(1,),
            object_long_sides=(0.25,),
            bbox_width_height_ratios=(target_ratio,),
            scene_scale_medians=(0.25,),
            within_scene_scale_ratios=(1.0,),
            background_colors=((230.0, 230.0, 230.0),),
            background_gradient_std=(1.0,),
            background_sensor_std=(1.0,),
            source_images=1,
            source_boxes=1,
        )
        transformed = _transform_cutout(
            cutout,
            spec,
            random.Random(31),
            scene_scale=0.25,
            calibration=calibration,
        )
        box = mask_bbox(transformed.mask)
        assert box is not None
        assert max(box[2], box[3]) == pytest.approx(32, abs=1)
        assert box[2] / box[3] == pytest.approx(target_ratio, abs=0.12)


def test_geometry_aware_selection_prefers_capable_cutout(tmp_path: Path) -> None:
    round_cutout = Cutout(
        "round", 0, "bean", tmp_path / "round.png", "round-source",
        intrinsic_aspect_ratio=1.10,
    )
    elongated_cutout = Cutout(
        "elongated", 0, "bean", tmp_path / "elongated.png", "long-source",
        intrinsic_aspect_ratio=2.20,
    )
    calibration = SceneCalibration(
        scene_counts=(1,),
        object_long_sides=(0.25,),
        bbox_width_height_ratios=(2.0,),
        scene_scale_medians=(0.25,),
        within_scene_scale_ratios=(1.0,),
        background_colors=((230.0, 230.0, 230.0),),
        background_gradient_std=(1.0,),
        background_sensor_std=(1.0,),
        source_images=1,
        source_boxes=1,
        bbox_width_height_ratios_by_class={0: (2.0,)},
    )

    chosen, targets, repeated, fallbacks = _choose_unique_cutouts(
        [round_cutout, elongated_cutout],
        1,
        CompositionSpec(object_range=(1, 1)),
        random.Random(5),
        calibration=calibration,
        geometry_rng=random.Random(7),
    )

    assert chosen == [elongated_cutout]
    assert targets == [2.0]
    assert repeated == 0
    assert fallbacks == 0

    fallback_choice, _, _, fallback_count = _choose_unique_cutouts(
        [round_cutout],
        1,
        CompositionSpec(object_range=(1, 1)),
        random.Random(5),
        calibration=calibration,
        geometry_rng=random.Random(7),
    )
    assert fallback_choice == [round_cutout]
    assert fallback_count == 1


def test_geometry_match_precedes_parent_uniqueness(tmp_path: Path) -> None:
    cutouts = [
        Cutout(
            f"long-{class_id}",
            class_id,
            f"class-{class_id}",
            tmp_path / f"long-{class_id}.png",
            f"long-source-{class_id}",
            source_parent_id="shared-parent",
            intrinsic_aspect_ratio=2.2,
        )
        for class_id in (0, 1)
    ] + [
        Cutout(
            f"round-{class_id}",
            class_id,
            f"class-{class_id}",
            tmp_path / f"round-{class_id}.png",
            f"round-source-{class_id}",
            source_parent_id=f"unique-parent-{class_id}",
            intrinsic_aspect_ratio=1.1,
        )
        for class_id in (0, 1)
    ]
    calibration = SceneCalibration(
        scene_counts=(2,),
        object_long_sides=(0.25,),
        bbox_width_height_ratios=(2.0,),
        scene_scale_medians=(0.25,),
        within_scene_scale_ratios=(1.0,),
        background_colors=((230.0, 230.0, 230.0),),
        background_gradient_std=(1.0,),
        background_sensor_std=(1.0,),
        source_images=1,
        source_boxes=2,
        bbox_width_height_ratios_by_class={0: (2.0,), 1: (2.0,)},
    )

    chosen, _, repeated, fallbacks = _choose_unique_cutouts(
        cutouts,
        2,
        CompositionSpec(object_range=(2, 2)),
        random.Random(13),
        calibration=calibration,
        geometry_rng=random.Random(17),
    )

    assert {item.asset_id for item in chosen} == {"long-0", "long-1"}
    assert repeated == 0
    assert fallbacks == 0


def test_object_library_uses_train_and_excludes_test(tmp_path: Path) -> None:
    source = tmp_path / "classification"
    output = tmp_path / "library"
    _write_classification_assets(source)

    result = prepare_classification_library(source, output)
    manifest_path = output / "object_library.json"
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in legacy_manifest["assets"]:
        row.pop("intrinsic_aspect_ratio", None)
    manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
    upgrade = backfill_intrinsic_geometry(output, progress_every=2)
    repeated_upgrade = backfill_intrinsic_geometry(output, progress_every=2)
    classes, cutouts, info = load_object_library(output)

    assert upgrade == {"assets": 4, "profiled": 4, "remaining": 0}
    assert repeated_upgrade == {"assets": 4, "profiled": 0, "remaining": 0}
    assert set(classes.values()) == {"normal", "defect"}
    assert len(cutouts) == 4
    assert all(item.intrinsic_aspect_ratio is not None for item in cutouts)
    assert {item.source_split for item in cutouts} == {"train"}
    assert result["source"]["skipped_by_split"] == {"test": 4}
    assert info["rejected_splits"] == {}


def test_yolo_library_hashes_only_reservoir_selected_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "yolo"
    output = tmp_path / "library"
    _write_yolo_cutout_source(source, images_per_class=8)
    original_hash = library_module.image_sha256
    hashed_paths = []

    def counted_hash(path: Path) -> str:
        hashed_paths.append(path)
        return original_hash(path)

    monkeypatch.setattr(library_module, "image_sha256", counted_hash)
    result = prepare_yolo_library(
        source,
        output,
        max_assets_per_class=1,
        candidate_multiplier=1,
        max_assets_per_image_class=1,
        box_padding_fraction=0.12,
        seed=9,
    )

    assert result["audit"]["assets_by_class"] == {"defect": 1, "normal": 1}
    assert len(hashed_paths) == 2
    index = result["source"]["index"]
    assert index["images_indexed"] == 16
    assert index["sampled_candidates_by_class"] == {"defect": 1, "normal": 1}


def test_visibility_compositor_is_deterministic_and_masks_are_consistent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "classification"
    library = tmp_path / "library"
    _write_classification_assets(source)
    prepare_classification_library(source, library)
    _, cutouts, _ = load_object_library(library)
    spec = CompositionSpec(
        canvas_size=(128, 128),
        object_range=(5, 5),
        object_scale=(0.20, 0.28),
        mode="visibility",
        use_shadows=False,
    )
    background = Image.new("RGB", spec.canvas_size, "white")

    first = compose_scene(background, cutouts, spec, random.Random(19))
    second = compose_scene(background, cutouts, spec, random.Random(19))

    assert first.image.tobytes() == second.image.tobytes()
    assert first.target_visibility_bin == second.target_visibility_bin
    assert [item.visibility_ratio for item in first.instances] == [
        item.visibility_ratio for item in second.instances
    ]
    for item in first.instances:
        assert item.visible_mask is not None
        assert not np.any(item.visible_mask & ~item.full_mask)
        expected = float(item.visible_mask.sum()) / float(item.full_mask.sum())
        assert abs(expected - item.visibility_ratio) < 1e-12


def test_visibility_compositor_handles_two_object_extreme_scene(tmp_path: Path) -> None:
    source = tmp_path / "classification"
    library = tmp_path / "library"
    _write_classification_assets(source)
    prepare_classification_library(source, library)
    _, cutouts, _ = load_object_library(library)
    spec = CompositionSpec(
        canvas_size=(128, 128),
        object_range=(2, 2),
        object_scale=(0.20, 0.24),
        mode="visibility",
        target_bin_weights={"extreme": 1.0},
        use_shadows=False,
    )

    scene = compose_scene(
        Image.new("RGB", spec.canvas_size, "white"),
        cutouts,
        spec,
        random.Random(3),
    )

    assert len(scene.instances) == 2
    assert scene.controlled_instances == 1
    assert scene.target_visibility_bin == "extreme"


def test_naive_and_visibility_arms_share_assets_and_transform_scales(
    tmp_path: Path,
) -> None:
    source = tmp_path / "classification"
    library = tmp_path / "library"
    _write_classification_assets(source)
    prepare_classification_library(source, library)
    _, cutouts, _ = load_object_library(library)
    base = CompositionSpec(
        canvas_size=(128, 128),
        object_range=(4, 4),
        object_scale=(0.18, 0.24),
        mode="naive",
        use_shadows=False,
    )
    background = Image.new("RGB", base.canvas_size, "white")
    calibration = SceneCalibration(
        scene_counts=(4,),
        object_long_sides=(0.22,),
        bbox_width_height_ratios=(1.4,),
        scene_scale_medians=(0.22,),
        within_scene_scale_ratios=(1.0,),
        background_colors=((230.0, 230.0, 230.0),),
        background_gradient_std=(1.0,),
        background_sensor_std=(1.0,),
        source_images=1,
        source_boxes=4,
        bbox_width_height_ratios_by_class={0: (1.4,), 1: (1.4,)},
    )

    naive = compose_scene(
        background, cutouts, base, random.Random(23), calibration=calibration
    )
    visibility = compose_scene(
        background,
        cutouts,
        replace(base, mode="visibility"),
        random.Random(23),
        calibration=calibration,
    )

    assert [item.cutout.asset_id for item in naive.instances] == [
        item.cutout.asset_id for item in visibility.instances
    ]
    assert [int(item.full_mask.sum()) for item in naive.instances] == [
        int(item.full_mask.sum()) for item in visibility.instances
    ]
    assert [mask_bbox(item.full_mask)[2:] for item in naive.instances] == [
        mask_bbox(item.full_mask)[2:] for item in visibility.instances
    ]
    assert [item.target_bbox_ratio for item in naive.instances] == [
        item.target_bbox_ratio for item in visibility.instances
    ]
    assert all(item.target_bbox_ratio == pytest.approx(1.4) for item in naive.instances)
    assert naive.scene_mode == visibility.scene_mode


def test_generate_and_audit_vadcp_dataset(tmp_path: Path) -> None:
    source = tmp_path / "classification"
    library = tmp_path / "library"
    real = tmp_path / "real"
    output = tmp_path / "vadcp"
    _write_classification_assets(source)
    _write_real_dataset(real)
    prepare_classification_library(source, library)

    manifest = generate_vadcp_dataset(
        real,
        library,
        output,
        synthetic_images=3,
        seed=7,
        mode="visibility",
        canvas_size=128,
        object_range=(4, 6),
        object_scale=(0.18, 0.28),
        use_shadows=False,
    )
    audit = audit_vadcp_dataset(output)
    visual = run_vadcp_visual_audit(
        output, tmp_path / "visual", samples=2, seed=7
    )

    assert manifest["synthetic_images"] == 3
    assert manifest["geometry_target_hit_rate"] is not None
    assert manifest["real_images"] == {"train": 1, "val": 1, "test": 1}
    assert audit["safe_for_training"], audit["errors"]
    assert isinstance(audit["geometry_ready"], bool)
    assert audit["synthetic_images"] == 3
    assert visual["samples"] == 2
    assert Path(visual["contact_sheet"]).is_file()
    assert Path(visual["raw_contact_sheet"]).is_file()
    metadata = json.loads(
        (output / "metadata" / "instances_synthetic_train.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["annotations"]
    assert all(
        row["source_split"] in {"train", "unspecified"}
        for row in metadata["annotations"]
    )


def test_calibrated_physics_scene_and_realism_audit(tmp_path: Path) -> None:
    source = tmp_path / "classification"
    library = tmp_path / "library"
    real = tmp_path / "real"
    output = tmp_path / "vadcp"
    _write_classification_assets(source)
    _write_real_dataset(real)
    prepare_classification_library(source, library)
    calibration = build_scene_calibration(
        discover_layout(real), seed=5, background_samples=2
    )
    profile_path = save_scene_calibration(calibration, tmp_path / "profile.json")
    restored = load_scene_calibration(profile_path)

    manifest = generate_vadcp_dataset(
        real,
        library,
        output,
        synthetic_images=4,
        seed=5,
        mode="visibility",
        canvas_size=128,
        object_range=(4, 6),
        include_real_train=False,
        scene_profile=restored,
    )
    cutout_visual = run_cutout_visual_audit(
        library, tmp_path / "cutout-visual", samples=4, seed=5
    )
    reference = build_realism_reference(real)
    realism = audit_vadcp_realism(
        real,
        output,
        tmp_path / "realism.json",
        seed=5,
        real_profile=reference,
    )

    assert restored.source_images == 1
    assert restored.bbox_width_height_ratios == pytest.approx((0.8, 0.8))
    assert restored.bbox_width_height_ratios_by_class == {
        0: pytest.approx((0.8,)),
        1: pytest.approx((0.8,)),
    }
    assert manifest["scene_calibration"]["summary"]["source_boxes"] == 2
    assert sum(manifest["scene_modes"].values()) == 4
    assert manifest["repeated_assets"] >= 0
    assert Path(cutout_visual["contact_sheet"]).is_file()
    assert realism["synthetic_physics"]["generated_instances"] >= 16
    assert "labeled_density" in realism["comparisons"]


def test_visibility_evaluation_ignores_other_bins_and_counts_duplicates() -> None:
    annotations = [
        {
            "image_id": 1,
            "category_id": 0,
            "bbox": [0, 0, 10, 10],
            "visibility_bin": "clear",
            "ignore": 0,
        },
        {
            "image_id": 1,
            "category_id": 0,
            "bbox": [20, 0, 10, 10],
            "visibility_bin": "severe",
            "ignore": 0,
        },
    ]
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.95},
        {"image_id": 1, "category_id": 0, "bbox": [20, 0, 10, 10], "score": 0.90},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.80},
    ]

    visibility = evaluate_predictions_by_visibility(
        predictions, annotations, {0: "bean"}, iou_thresholds=(0.50,)
    )
    counts = count_metrics(predictions, annotations, [1], confidence=0.25)

    assert visibility["clear"]["ap50"] == 1.0
    clear = visibility["clear"]["per_class"]["bean"]
    assert clear["ignored_detections50"] == 1
    assert clear["duplicate_predictions50"] == 1
    assert counts["mean_count_bias"] == 1.0
    assert counts["duplicate_predictions"] == 1
    assert counts["duplicate_prediction_rate"] == 1 / 3
