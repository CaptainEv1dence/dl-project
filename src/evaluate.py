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


def _default_image_size(model_name: str, image_size: int | None) -> int:
    if image_size is not None:
        return image_size
    return 224 if model_name == "efficientnet_b2" else 48


def evaluate_checkpoint(
    csv_path,
    checkpoint_path,
    model_name: str,
    split: str = "test",
    output_dir="outputs/metrics",
    batch_size: int = 64,
    num_workers: int = 0,
    image_size: int | None = None,
    imagenet_norm: bool | None = None,
) -> dict:
    """Load a checkpoint, run inference on a split, compute metrics, and save artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = _default_image_size(model_name, image_size)
    if imagenet_norm is None:
        imagenet_norm = model_name == "efficientnet_b2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = {}
    if model_name in {"resnet18", "efficientnet_b2"}:
        # Do not download ImageNet weights for evaluation; checkpoint overwrites params.
        model_kwargs["weights"] = None
        model_kwargs["freeze_backbone"] = False

    model = create_model(model_name, **model_kwargs)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.to(device)
    model.eval()

    dataset = FER2013Dataset(
        csv_path,
        split=split,
        transform=get_eval_transforms(image_size=image_size, imagenet_norm=imagenet_norm),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
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

    all_labels = list(range(NUM_CLASSES))
    metrics = compute_classification_metrics(y_true, y_pred, labels=all_labels)
    metrics["split"] = split
    metrics["num_samples"] = len(y_true)
    metrics["model_name"] = model_name
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["image_size"] = image_size
    metrics["imagenet_norm"] = bool(imagenet_norm)

    report_path = output_dir / f"classification_report_{model_name}_{split}.json"
    report_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    class_names = [EMOTION_LABELS[i] for i in all_labels]
    cm_path = output_dir / f"confusion_matrix_{model_name}_{split}.png"
    save_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        cm_path,
        title=f"Confusion Matrix: {model_name} ({split})",
    )

    metrics["report_path"] = str(report_path)
    metrics["confusion_matrix_path"] = str(cm_path)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a dataset split.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="baseline_cnn")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default="outputs/metrics")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--imagenet-norm", action=argparse.BooleanOptionalAction, default=None)

    args = parser.parse_args()

    result = evaluate_checkpoint(
        args.csv,
        args.checkpoint,
        args.model,
        args.split,
        args.out,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        imagenet_norm=args.imagenet_norm,
    )

    print(f"accuracy: {result['accuracy']:.4f}")
    print(f"macro_f1: {result['macro_f1']:.4f}")
    print(f"weighted_f1: {result['weighted_f1']:.4f}")
    print(f"num_samples: {result['num_samples']}")
    print(f"report: {result.get('report_path')}")
    print(f"confusion_matrix: {result.get('confusion_matrix_path')}")
