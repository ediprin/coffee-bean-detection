from coffee_detector.analysis.igem_af2_targeted_rescue_audit import audit


def _payload(model, rows):
    events = {}
    for index, row in enumerate(rows):
        events[f"img.jpg::gt{index}"] = {
            "target_key": f"img.jpg::gt{index}",
            "image": "img.jpg",
            "target_index": index,
            "gt_class_id": row[0],
            "gt_class_name": row[1],
            "matched": row[2],
            "pred_class_name": row[3],
            "correct": row[4],
        }
    return {
        "protocol": "faruq-v3-validation-object-events-v1",
        "model": model,
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "events": events,
    }


def test_targeted_rescue_counts_classification_and_harm():
    igem = _payload(
        "IGEM1",
        [
            (0, "A", True, "B", False),
            (0, "A", True, "A", True),
            (1, "B", False, None, False),
            (1, "B", True, "A", False),
        ],
    )
    af2 = _payload(
        "AF2",
        [
            (0, "A", True, "A", True),
            (0, "A", True, "B", False),
            (1, "B", True, "B", True),
            (1, "B", True, "A", False),
        ],
    )

    result = audit(igem, af2)
    assert result["global"]["igem_errors_total"] == 3
    assert result["global"]["af2_rescues_total"] == 2
    assert result["global"]["igem_classification_errors_iou50"] == 2
    assert result["global"]["af2_classification_rescues_iou50"] == 1
    assert result["global"]["af2_harms_when_igem_correct"] == 1
    assert result["global"]["net_total_rescue_minus_harm"] == 1

    directed = {row["family"]: row for row in result["directed_confusion_families"]}
    assert directed["A -> B"]["support_igem_classification_errors"] == 1
    assert directed["A -> B"]["af2_rescues"] == 1
    assert directed["B -> A"]["support_igem_classification_errors"] == 1
    assert directed["B -> A"]["af2_rescues"] == 0

    undirected = {row["family"]: row for row in result["undirected_confusion_families"]}
    assert undirected["A <-> B"]["support_igem_classification_errors"] == 2
    assert undirected["A <-> B"]["af2_rescues"] == 1
