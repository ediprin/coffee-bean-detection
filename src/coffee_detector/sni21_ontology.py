from __future__ import annotations

from pathlib import Path

import yaml


SNI21_CLASSES = (
    "biji_berkulit_tanduk",
    "biji_berlubang_lebih_satu",
    "biji_berlubang_satu",
    "biji_bertutul_tutul",
    "biji_coklat",
    "biji_hitam",
    "biji_hitam_pecah",
    "biji_hitam_sebagian",
    "biji_muda",
    "biji_normal",
    "biji_pecah",
    "kopi_gelondong",
    "kulit_kopi_ukuran_besar",
    "kulit_kopi_ukuran_kecil",
    "kulit_kopi_ukuran_sedang",
    "kulit_tanduk_ukuran_besar",
    "kulit_tanduk_ukuran_kecil",
    "kulit_tanduk_ukuran_sedang",
    "tanah_batu_ranting_besar",
    "tanah_batu_ranting_kecil",
    "tanah_batu_ranting_sedang",
)

SNI_DEFECT_WEIGHTS = {
    "biji_hitam": 1.0,
    "biji_hitam_sebagian": 0.5,
    "biji_hitam_pecah": 0.5,
    "kopi_gelondong": 1.0,
    "biji_coklat": 0.25,
    "kulit_kopi_ukuran_besar": 1.0,
    "kulit_kopi_ukuran_sedang": 0.5,
    "kulit_kopi_ukuran_kecil": 0.2,
    "biji_berkulit_tanduk": 0.5,
    "kulit_tanduk_ukuran_besar": 0.5,
    "kulit_tanduk_ukuran_sedang": 0.2,
    "kulit_tanduk_ukuran_kecil": 0.1,
    "biji_pecah": 0.2,
    "biji_muda": 0.2,
    "biji_berlubang_satu": 0.1,
    "biji_berlubang_lebih_satu": 0.2,
    "biji_bertutul_tutul": 0.1,
    "tanah_batu_ranting_besar": 5.0,
    "tanah_batu_ranting_sedang": 2.0,
    "tanah_batu_ranting_kecil": 1.0,
    "biji_normal": 0.0,
}

SIGNATURE_FIELDS = (
    "entity_family",
    "primary_condition",
    "surface_extent",
    "integrity_fraction",
    "relative_completeness",
    "hole_count",
    "physical_size_mm",
)


def default_ontology_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs/sni21/structured_ontology_v1.yaml"


def validate_sni21_ontology(payload: dict) -> None:
    if payload.get("status") != "protocol_only_no_training_authorized":
        raise ValueError("Ontologi belum dikunci sebagai protocol-only")
    classes = payload.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("Field classes tidak valid")
    if set(classes) != set(SNI21_CLASSES):
        missing = sorted(set(SNI21_CLASSES) - set(classes))
        extra = sorted(set(classes) - set(SNI21_CLASSES))
        raise ValueError(f"Kelas ontologi tidak lengkap: missing={missing}, extra={extra}")

    signatures: dict[tuple, str] = {}
    for class_name in SNI21_CLASSES:
        row = classes[class_name]
        missing_fields = [field for field in SIGNATURE_FIELDS if field not in row]
        if missing_fields:
            raise ValueError(f"{class_name} kehilangan field {missing_fields}")
        observed_weight = float(row.get("defect_weight"))
        expected_weight = SNI_DEFECT_WEIGHTS[class_name]
        if observed_weight != expected_weight:
            raise ValueError(
                f"Bobot SNI {class_name} salah: {observed_weight} != {expected_weight}"
            )
        flags = row.get("positive_flags")
        if not isinstance(flags, list) or not flags:
            raise ValueError(f"positive_flags kosong: {class_name}")
        signature = tuple(row.get(field) for field in SIGNATURE_FIELDS)
        if signature in signatures:
            raise ValueError(
                f"Signature leaf tidak unik: {class_name} dan {signatures[signature]}"
            )
        signatures[signature] = class_name

    if classes["biji_hitam_pecah"]["positive_flags"] != ["black", "broken"]:
        raise ValueError("Biji hitam pecah harus mengawasi atribut black dan broken")
    if classes["tanah_batu_ranting_besar"]["physical_size_mm"] != "greater_than_10":
        raise ValueError("Ambang benda asing besar harus >10 mm")
    if payload.get("observability", {}).get("physical_size_mm") != "calibrated_scale_required":
        raise ValueError("Ukuran fisik harus memerlukan kalibrasi skala")


def load_sni21_ontology(path: str | Path | None = None) -> dict:
    source = Path(path).expanduser().resolve() if path else default_ontology_path()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    validate_sni21_ontology(payload)
    payload["source"] = str(source)
    return payload


def structured_target_for(class_name: str, payload: dict | None = None) -> dict:
    ontology = payload if payload is not None else load_sni21_ontology()
    validate_sni21_ontology(ontology)
    if class_name not in ontology["classes"]:
        raise KeyError(f"Kelas SNI-21 tidak dikenal: {class_name}")
    row = ontology["classes"][class_name]
    return {
        "original_class": class_name,
        "entity_family": row["entity_family"],
        "primary_condition": row["primary_condition"],
        "positive_flags": tuple(row["positive_flags"]),
        "explicit_negative_flags": tuple(row.get("explicit_negative_flags", [])),
        "observed_attributes": {
            field: row[field]
            for field in SIGNATURE_FIELDS[2:]
            if row[field] is not None
        },
        "defect_weight": float(row["defect_weight"]),
    }
