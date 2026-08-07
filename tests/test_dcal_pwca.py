import json
from pathlib import Path

import torch
import yaml

from coffee_detector.dcal_pwca import (
    DCALPWCAConfig,
    DCALPWCADetectHead,
    DCALPWCADetectionModel,
    P5CrossAttentionRegularizer,
    load_dcal_pwca_weights,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(mode: str = "pwca", nc: int = 5):
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(23)
    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = DCALPWCADetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        dcal_pwca=DCALPWCAConfig(mode=mode, hidden_dim=16, num_heads=4, mlp_ratio=2.0),
    ).eval()
    load_dcal_pwca_weights(candidate, source)
    return source, candidate


def test_inference_delegates_to_native_and_never_runs_regularizer():
    source, candidate = _models("pwca")
    head = candidate.model[-1]
    assert isinstance(head, DCALPWCADetectHead)
    original = head.regularizer.forward

    def forbidden(*args, **kwargs):
        raise AssertionError("PWCA must be removed at inference")

    head.regularizer.forward = forbidden
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        native = source(image)
        transferred = candidate(image)
    head.regularizer.forward = original
    assert torch.equal(transferred[0], native[0])
    assert torch.equal(transferred[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"])
    assert torch.equal(transferred[1]["one2one"]["scores"], native[1]["one2one"]["scores"])


def test_identity_start_preserves_training_outputs_and_one2one_is_native():
    source, candidate = _models("pwca")
    source.train(); candidate.train()
    image = torch.randn(2, 3, 128, 128)
    native = source(image)
    transferred = candidate(image)
    for branch in ("one2many", "one2one"):
        assert torch.equal(transferred[branch]["boxes"], native[branch]["boxes"])
        assert torch.equal(transferred[branch]["scores"], native[branch]["scores"])


def test_nonzero_correction_changes_only_p5_one2many_scores():
    source, candidate = _models("pwca")
    source.train(); candidate.train()
    head = candidate.model[-1]
    with torch.no_grad():
        head.regularizer.classifier.bias[0] = 0.5
    image = torch.randn(2, 3, 128, 128)
    native = source(image)
    transferred = candidate(image)
    assert torch.equal(transferred["one2many"]["boxes"], native["one2many"]["boxes"])
    assert torch.equal(transferred["one2one"]["boxes"], native["one2one"]["boxes"])
    assert torch.equal(transferred["one2one"]["scores"], native["one2one"]["scores"])

    native_scores = native["one2many"]["scores"]
    candidate_scores = transferred["one2many"]["scores"]
    level_sizes = [feature.shape[-2] * feature.shape[-1] for feature in transferred["one2many"]["feats"]]
    p5_start = level_sizes[0] + level_sizes[1]
    assert torch.equal(candidate_scores[:, :, :p5_start], native_scores[:, :, :p5_start])
    assert not torch.equal(candidate_scores[:, :, p5_start:], native_scores[:, :, p5_start:])


def test_sa0_uses_n_kv_tokens_and_pw1_uses_2n_for_batch_gt1():
    feature = torch.randn(3, 24, 8, 8)
    sa = P5CrossAttentionRegularizer(24, 5, DCALPWCAConfig(mode="sa", hidden_dim=16, num_heads=4))
    pw = P5CrossAttentionRegularizer(24, 5, DCALPWCAConfig(mode="pwca", hidden_dim=16, num_heads=4))
    sa.forward_features(feature)
    torch.manual_seed(7)
    pw.forward_features(feature)
    assert sa.last_query_tokens == 64 and sa.last_kv_tokens == 64
    assert pw.last_query_tokens == 64 and pw.last_kv_tokens == 128
    assert pw.last_pair_offset in {1, 2}


def test_pwca_target_features_depend_on_paired_image_but_sa_does_not():
    torch.manual_seed(31)
    sa = P5CrossAttentionRegularizer(8, 3, DCALPWCAConfig(mode="sa", hidden_dim=8, num_heads=2)).eval()
    pw = P5CrossAttentionRegularizer(8, 3, DCALPWCAConfig(mode="pwca", hidden_dim=8, num_heads=2)).eval()
    pw.load_state_dict(sa.state_dict(), strict=True)
    first = torch.randn(1, 8, 4, 4)
    second_a = torch.randn(1, 8, 4, 4)
    second_b = second_a.clone()
    second_b[:, 0] = second_b[:, 0] + 5.0  # survives per-token LayerNorm
    batch_a = torch.cat((first, second_a), dim=0)
    batch_b = torch.cat((first, second_b), dim=0)

    with torch.inference_mode():
        sa_a = sa.forward_features(batch_a)[0]
        sa_b = sa.forward_features(batch_b)[0]
        torch.manual_seed(1)
        pw_a = pw.forward_features(batch_a)[0]
        torch.manual_seed(1)
        pw_b = pw.forward_features(batch_b)[0]
    assert torch.allclose(sa_a, sa_b, atol=1e-6, rtol=0.0)
    assert not torch.allclose(pw_a, pw_b, atol=1e-5, rtol=0.0)


def test_batch_one_pwca_falls_back_to_self_attention_length():
    module = P5CrossAttentionRegularizer(8, 3, DCALPWCAConfig(mode="pwca", hidden_dim=8, num_heads=2))
    module.forward_features(torch.randn(1, 8, 4, 4))
    assert module.last_query_tokens == module.last_kv_tokens == 16
    assert module.last_pair_offset == 0


def test_configs_are_capacity_matched_except_attention_mode():
    sa = yaml.safe_load((ROOT / "configs/dcal_pwca/SA0_p5_self_attention_control.yaml").read_text())
    pw = yaml.safe_load((ROOT / "configs/dcal_pwca/PW1_p5_pairwise_cross_attention.yaml").read_text())
    assert sa["dcal_pwca"]["mode"] == "sa"
    assert pw["dcal_pwca"]["mode"] == "pwca"
    for key in ("hidden_dim", "num_heads", "mlp_ratio", "correction_scale"):
        assert sa["dcal_pwca"][key] == pw["dcal_pwca"][key]
    assert sa["train"] == pw["train"]


def test_notebook_is_branch_correct_and_val_only():
    path = ROOT / "notebooks/Faruq_V3_DCAL_PWCA_Screening_Colab.ipynb"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
    assert "agent/dcal-pwca-screening" in source
    assert "run_faruq_v3_dcal_pwca_screening" in source
    assert "--authorize-training" in source
    assert "split=test" not in source.lower()
