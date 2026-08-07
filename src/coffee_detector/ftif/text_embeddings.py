from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml


def load_prompt_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload.get("base_prompt"), str) or not payload["base_prompt"].strip():
        raise ValueError("FTIF prompt manifest memerlukan base_prompt")
    classes = payload.get("classes")
    if not isinstance(classes, dict) or len(classes) != 21:
        raise ValueError("FTIF prompt manifest harus memiliki tepat 21 kelas")
    if any(not isinstance(value, str) or not value.strip() for value in classes.values()):
        raise ValueError("Semua specific prompt harus non-kosong")
    return payload


def prompt_manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _normalize_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(int(k) for k in names)]
    return [str(value) for value in names]


def validate_manifest_against_class_names(manifest: dict[str, Any], class_names: Any) -> list[str]:
    names = _normalize_names(class_names)
    manifest_names = list(manifest["classes"])
    if names != manifest_names:
        raise ValueError(
            "Urutan nama kelas dataset harus identik dengan prompt manifest. "
            f"dataset={names}, manifest={manifest_names}"
        )
    return names


def build_prompt_texts(manifest: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    names = list(manifest["classes"])
    base = str(manifest["base_prompt"]).strip()
    specific = [str(manifest["classes"][name]).strip() for name in names]
    base_prompts = [base for _ in names]
    return names, base_prompts, specific


def generate_clip_text_embeddings(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cpu",
) -> dict[str, Any]:
    """Generate and cache frozen text embeddings for FTIF.

    The LFDet paper states that its CLIP text encoder is frozen but the parsed
    implementation text available to this project does not identify a concrete
    CLIP variant. `ViT-B-32` with OpenAI weights is therefore an explicit
    transfer choice, not a claim about the paper implementation.

    `open_clip_torch` is imported lazily so repository CI never needs network
    access or this optional dependency. Embeddings are kept *unnormalized*;
    the paper says base and specific embeddings are added, so the combined
    vector is stored as the direct sum. Alignment later performs cosine
    normalization explicitly as in LFDet Eq. (19).
    """
    try:
        import open_clip
    except ImportError as exc:  # pragma: no cover - optional Colab dependency
        raise RuntimeError(
            "Install optional dependency `open_clip_torch` sebelum membuat FTIF embeddings"
        ) from exc

    manifest = load_prompt_manifest(manifest_path)
    names, base_prompts, specific_prompts = build_prompt_texts(manifest)
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with torch.inference_mode():
        base_tokens = tokenizer(base_prompts).to(device)
        specific_tokens = tokenizer(specific_prompts).to(device)
        base_embeddings = model.encode_text(base_tokens, normalize=False).float().cpu()
        specific_embeddings = model.encode_text(specific_tokens, normalize=False).float().cpu()
        combined_embeddings = base_embeddings + specific_embeddings

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "faruq-v3-ftif-text-embeddings-v1",
        "class_names": names,
        "base_prompt": manifest["base_prompt"],
        "specific_prompts": specific_prompts,
        "base_embeddings": base_embeddings,
        "specific_embeddings": specific_embeddings,
        "combined_embeddings": combined_embeddings,
        "encoder": {
            "library": "open_clip_torch",
            "model_name": model_name,
            "pretrained": pretrained,
            "frozen": True,
            "transfer_choice": True,
        },
        "manifest_sha256": prompt_manifest_sha256(manifest_path),
    }
    torch.save(payload, output)
    metadata = {
        "output": str(output),
        "class_names": names,
        "embedding_dim": int(specific_embeddings.shape[1]),
        "encoder": payload["encoder"],
        "manifest_sha256": payload["manifest_sha256"],
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def load_text_embedding_payload(
    path: str | Path,
    *,
    class_names: Any | None = None,
    prompt_mode: str = "base_specific",
) -> tuple[torch.Tensor, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("format") != "faruq-v3-ftif-text-embeddings-v1":
        raise ValueError("Format FTIF embedding cache tidak dikenali")
    names = [str(value) for value in payload["class_names"]]
    if class_names is not None and names != _normalize_names(class_names):
        raise ValueError("Urutan kelas embedding cache berbeda dari dataset")
    if prompt_mode == "specific":
        key = "specific_embeddings"
    elif prompt_mode == "base_specific":
        key = "combined_embeddings"
    else:
        raise ValueError("prompt_mode harus specific atau base_specific")
    embeddings = payload[key].float().contiguous()
    if embeddings.ndim != 2 or embeddings.shape[0] != len(names):
        raise ValueError("Shape FTIF text embeddings tidak valid")
    metadata = {
        "class_names": names,
        "encoder": payload.get("encoder", {}),
        "manifest_sha256": payload.get("manifest_sha256"),
        "prompt_mode": prompt_mode,
        "embedding_dim": int(embeddings.shape[1]),
    }
    return embeddings, metadata
