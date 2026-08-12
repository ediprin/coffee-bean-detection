import json
from pathlib import Path

import torch
import yaml

from coffee_detector.ftif import (
    FTIFConfig,
    FTIFDetectionModel,
    FTIFDetectHead,
    bidirectional_alignment_loss,
    build_prompt_texts,
    load_ftif_detector_weights,
    load_prompt_manifest,
    load_text_embedding_payload,
    validate_manifest_against_class_names,
)
from coffee_detector.experiments.run_faruq_v3_ftif_screening import _run_is_complete

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
PROMPTS = ROOT / "configs/ftif/sni21_prompts.yaml"
ONTOLOGY = ROOT / "configs/sni21/structured_ontology_v1.yaml"


def _models(nc: int = 5, *, alignment: bool = False):
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(19)
    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    text = torch.randn(nc, 16)
    candidate = FTIFDetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        ftif=FTIFConfig(
            prompt_mode="base_specific",
            bidirectional_alignment=alignment,
            temperature=0.07,
        ),
        text_embeddings=text,
    ).eval()
    load_ftif_detector_weights(candidate, source)
    return source, candidate


def test_ftif_identity_start_preserves_native_inference_boxes_and_scores():
    source, candidate = _models()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        source_output = source(image)
        candidate_output = candidate(image)
    assert torch.equal(candidate_output[0], source_output[0])
    assert torch.equal(candidate_output[1]["one2one"]["boxes"], source_output[1]["one2one"]["boxes"])
    assert torch.equal(candidate_output[1]["one2one"]["scores"], source_output[1]["one2one"]["scores"])


def test_ftif_box_branch_stays_native_when_text_correction_becomes_nonzero():
    source, candidate = _models()
    head = candidate.model[-1]
    assert isinstance(head, FTIFDetectHead)
    with torch.no_grad():
        head.integrators[0].logit_correction.bias[0] = 0.25
    source.train()
    candidate.train()
    image = torch.randn(1, 3, 128, 128)
    source_output = source(image)
    candidate_output = candidate(image)
    for branch in ("one2many", "one2one"):
        assert torch.equal(candidate_output[branch]["boxes"], source_output[branch]["boxes"])
    assert not torch.equal(candidate_output["one2many"]["scores"], source_output["one2many"]["scores"])


def test_ftif_similarity_exists_only_for_one2many_when_alignment_enabled():
    _, candidate = _models(alignment=True)
    candidate.train()
    output = candidate(torch.randn(2, 3, 128, 128))
    assert "ftif_similarity" in output["one2many"]
    assert "ftif_similarity" not in output["one2one"]
    scores = output["one2many"]["scores"].transpose(1, 2)
    similarity = output["one2many"]["ftif_similarity"]
    assert similarity.shape == scores.shape
    assert candidate.model[-1].config.temperature == 0.07


def test_ftif_e2e_loss_routes_alignment_only_to_one2many():
    _, candidate = _models(alignment=True)
    candidate.args = type(
        "Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 2}
    )()
    criterion = candidate.init_criterion()
    assert criterion.one2many.apply_ftif_alignment is True
    assert criterion.one2one.apply_ftif_alignment is False

    candidate.train()
    predictions = candidate(torch.randn(1, 3, 128, 128))
    batch = {
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    loss, components = criterion(predictions, batch)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(components).all()


def test_ftif_runner_does_not_treat_partial_best_as_complete(tmp_path):
    run_dir = tmp_path / "FT3_seed42"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    torch.save({"epoch": 0, "optimizer": {"state": {}}}, weights / "best.pt")
    assert _run_is_complete(run_dir, expected_epochs=50) is False

    torch.save({"epoch": 4, "optimizer": {"state": {}}}, weights / "last.pt")
    (run_dir / "results.csv").write_text(
        "epoch,metrics/mAP50-95(B)\n1,0.7\n2,0.8\n3,0.75\n4,0.76\n5,0.77\n",
        encoding="utf-8",
    )
    assert _run_is_complete(run_dir, expected_epochs=50) is False

    torch.save({"epoch": -1, "optimizer": None}, weights / "last.pt")
    assert _run_is_complete(run_dir, expected_epochs=50) is True


def test_bidirectional_alignment_prefers_correct_pairing():
    fg = torch.tensor([[True, True, False]])
    labels = torch.tensor([[0, 1, 0]])
    correct = torch.tensor([[[8.0, -8.0], [-8.0, 8.0], [-8.0, -8.0]]])
    swapped = torch.tensor([[[-8.0, 8.0], [8.0, -8.0], [-8.0, -8.0]]])
    loss_good, terms = bidirectional_alignment_loss(correct, fg, labels)
    loss_bad, _ = bidirectional_alignment_loss(swapped, fg, labels)
    assert loss_good < loss_bad
    assert set(terms) == {"positive_i2t", "negative_i2t", "positive_t2i", "negative_t2i"}


def test_prompt_manifest_matches_frozen_sni21_ontology_order():
    manifest = load_prompt_manifest(PROMPTS)
    ontology = yaml.safe_load(ONTOLOGY.read_text(encoding="utf-8"))
    names = list(ontology["classes"])
    assert validate_manifest_against_class_names(manifest, names) == names
    _, base_prompts, specific = build_prompt_texts(manifest)
    assert len(base_prompts) == len(specific) == 21
    text = "\n".join(specific).lower()
    assert "validation" not in text and "confusion" not in text


def test_embedding_cache_selects_specific_or_added_base_specific(tmp_path):
    names = list(load_prompt_manifest(PROMPTS)["classes"])
    base = torch.full((21, 4), 2.0)
    specific = torch.arange(84, dtype=torch.float32).reshape(21, 4)
    payload = {
        "format": "faruq-v3-ftif-text-embeddings-v1",
        "class_names": names,
        "base_embeddings": base,
        "specific_embeddings": specific,
        "combined_embeddings": base + specific,
        "encoder": {"model_name": "synthetic", "frozen": True},
        "manifest_sha256": "synthetic",
    }
    path = tmp_path / "text.pt"
    torch.save(payload, path)
    s, _ = load_text_embedding_payload(path, class_names=names, prompt_mode="specific")
    c, _ = load_text_embedding_payload(path, class_names=names, prompt_mode="base_specific")
    assert torch.equal(s, specific)
    assert torch.equal(c, base + specific)


def test_configs_encode_predeclared_three_arm_ablation():
    expected = {
        "FT1_specific_crossattn.yaml": ("specific", False),
        "FT2_base_specific_crossattn.yaml": ("base_specific", False),
        "FT3_base_specific_bidirectional.yaml": ("base_specific", True),
    }
    for filename, (prompt_mode, align) in expected.items():
        payload = yaml.safe_load((ROOT / "configs/ftif" / filename).read_text(encoding="utf-8"))
        assert payload["ftif"]["prompt_mode"] == prompt_mode
        assert payload["ftif"]["bidirectional_alignment"] is align
        assert payload["ftif"]["temperature"] == 0.07
        assert payload["ftif"]["alignment_weight"] == 1.0
        assert payload["train"]["epochs"] == 50
        assert payload["train"]["imgsz"] == 640


def test_notebook_is_branch_correct_val_only_and_generates_frozen_text_cache():
    notebook = ROOT / "notebooks/Faruq_V3_LFDet_FTIF_Screening_Colab.ipynb"
    assert notebook.is_file()
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
    assert "agent/lfdet-ftif-text-image-screening" in source
    assert "generate_clip_text_embeddings" in source
    assert "ViT-B-32" in source and "openai" in source
    assert "run_faruq_v3_ftif_screening" in source
    assert "--authorize-training" in source
    assert "CHUNK3_OUTPUT" in source
    assert "saved_epochs" in source
    assert "RESUME OUTPUT" in source
    assert "split=test" not in source.lower()
