from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import EMOTION_LABELS, NUM_CLASSES
from src.data.transforms import get_eval_transforms
from src.models import create_model
from src.utils.checkpoints import load_checkpoint


def _default_image_size(model_name: str, image_size: int | None) -> int:
    if image_size is not None:
        return image_size
    return 224 if model_name == "efficientnet_b2" else 48


def preprocess_image(
    image_path,
    mean: float = 0.5,
    std: float = 0.5,
    image_size: int = 48,
    imagenet_norm: bool = False,
) -> torch.Tensor:
    """Load an image from disk and return a model-ready tensor with batch dimension."""

    img = Image.open(image_path).convert("L")
    img = img.resize((48, 48), Image.BILINEAR)
    img_array = np.array(img, dtype=np.uint8)

    transform = get_eval_transforms(
        mean=mean,
        std=std,
        image_size=image_size,
        imagenet_norm=imagenet_norm,
    )
    tensor = transform(img_array)
    return tensor.unsqueeze(0)


def predict_image(
    checkpoint_path,
    image_path,
    model_name: str,
    image_size: int | None = None,
    imagenet_norm: bool | None = None,
) -> dict:
    """Load a checkpoint and run single-image inference."""

    image_size = _default_image_size(model_name, image_size)
    if imagenet_norm is None:
        imagenet_norm = model_name == "efficientnet_b2"

    model_kwargs = {}
    if model_name in {"resnet18", "efficientnet_b2"}:
        model_kwargs["weights"] = None
        model_kwargs["freeze_backbone"] = False

    model = create_model(model_name, **model_kwargs)
    load_checkpoint(checkpoint_path, model, map_location="cpu")
    model.eval()

    tensor = preprocess_image(
        image_path,
        image_size=image_size,
        imagenet_norm=imagenet_norm,
    )

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    label_id = int(probs.argmax())
    return {
        "label_id": label_id,
        "label": EMOTION_LABELS[label_id],
        "probabilities": {EMOTION_LABELS[i]: float(probs[i]) for i in range(NUM_CLASSES)},
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run single-image emotion prediction.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="baseline_cnn")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--imagenet-norm", action=argparse.BooleanOptionalAction, default=None)

    args = parser.parse_args()

    result = predict_image(
        args.checkpoint,
        args.image,
        args.model,
        image_size=args.image_size,
        imagenet_norm=args.imagenet_norm,
    )
    print(f"Predicted emotion: {result['label']} (label_id={result['label_id']})")
    print("Probabilities:")
    for emotion, prob in result["probabilities"].items():
        print(f" {emotion}: {prob:.4f}")
