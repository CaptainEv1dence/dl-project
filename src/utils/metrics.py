from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    labels: list[int] | None = None,
) -> dict:
    """
    Returns a JSON-serializable dict with keys:
    - "accuracy": float
    - "macro_f1": float
    - "weighted_f1": float
    - "per_class": dict (from classification_report output_dict=True, zero_division=0)
    - "confusion_matrix": list[list[int]]  (nested Python lists, NOT numpy array)
    """
    accuracy = accuracy_score(y_true, y_pred)

    f1_kwargs = {"zero_division": 0}
    if labels is not None:
        f1_kwargs["labels"] = labels

    macro_f1 = f1_score(y_true, y_pred, average="macro", **f1_kwargs)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", **f1_kwargs)

    per_class = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    cm_kwargs = {}
    if labels is not None:
        cm_kwargs["labels"] = labels

    cm = confusion_matrix(y_true, y_pred, **cm_kwargs)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
