from src.utils.metrics import compute_classification_metrics


def test_compute_classification_metrics_returns_required_keys():
    y_true = [0, 1, 1, 2]
    y_pred = [0, 1, 2, 2]

    metrics = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2])

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
