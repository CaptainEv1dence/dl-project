import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import EMOTION_LABELS, NUM_CLASSES
from src.data.dataset import FER2013Dataset
from src.data.transforms import get_eval_transforms
from src.models import create_model
from src.utils.checkpoints import load_checkpoint
from src.utils.metrics import compute_classification_metrics
from src.utils.plotting import save_confusion_matrix


def evaluate_checkpoint(
    csv_path,
    checkpoint_path,
    model_name: str,
    split: str = "test",
    output_dir="outputs/metrics",
) -> dict:
    """
    Load a checkpoint, run inference on a dataset split, compute metrics,
    and save the classification report JSON and confusion matrix PNG.

    Returns the metrics dict (JSON-serializable).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(model_name)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.to(device)
    model.eval()

    dataset = FER2013Dataset(csv_path, split=split, transform=get_eval_transforms())
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    if not y_true:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "split": split,
            "num_samples": 0,
        }

    # Always compute metrics over all 7 classes for a consistent confusion matrix shape.
    all_labels = list(range(NUM_CLASSES))
    metrics = compute_classification_metrics(y_true, y_pred, labels=all_labels)
    metrics["split"] = split
    metrics["num_samples"] = len(y_true)

    # Save JSON report
    report_path = output_dir / f"classification_report_{split}.json"
    report_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save confusion matrix PNG
    class_names = [EMOTION_LABELS[i] for i in all_labels]
    cm_path = output_dir / f"confusion_matrix_{split}.png"
    save_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        cm_path,
        title=f"Confusion Matrix ({split})",
    )

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a dataset split.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="baseline_cnn")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default="outputs/metrics")
    args = parser.parse_args()

    result = evaluate_checkpoint(args.csv, args.checkpoint, args.model, args.split, args.out)
    print(f"accuracy:    {result['accuracy']:.4f}")
    print(f"macro_f1:    {result['macro_f1']:.4f}")
    print(f"weighted_f1: {result['weighted_f1']:.4f}")
    print(f"num_samples: {result['num_samples']}")
