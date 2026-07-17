from coffee_detector.run_screening import _metric


def test_metric_distinguishes_map50_from_map50_95() -> None:
    metrics = {
        "metrics/mAP50-95(B)": 0.41,
        "metrics/mAP50(B)": 0.73,
    }

    assert _metric(metrics, "mAP50-95") == 0.41
    assert _metric(metrics, "mAP50") == 0.73

