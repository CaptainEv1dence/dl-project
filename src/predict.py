from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import EMOTION_LABELS, NUM_CLASSES
from src.data.transforms import get_eval_transforms
from src.models import create_model
from src.utils.checkpoints import load_checkpoint


def preprocess_image(image_path, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """
    Load an image from disk, convert to grayscale, resize to 48x48,
    normalise, and return a tensor of shape (1, 1, 48, 48).
    """
    img = Image.open(image_path).convert("L")  # grayscale
    img = img.resize((48, 48), Image.BILINEAR)
    img_array = np.array(img, dtype=np.uint8)  # shape (48, 48)

    transform = get_eval_transforms(mean=mean, std=std)
    tensor = transform(img_array)  # (1, 48, 48)
    return tensor.unsqueeze(0)  # (1, 1, 48, 48)


def predict_image(checkpoint_path, image_path, model_name: str) -> dict:
    """
    Load a checkpoint and run single-image inference.

    Returns:
        {
            "label_id": int,
            "label": str,
            "probabilities": {"anger": float, ...},
        }
    """
    model = create_model(model_name)
    load_checkpoint(checkpoint_path, model, map_location="cpu")
    model.eval()

    tensor = preprocess_image(image_path)  # (1, 1, 48, 48)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]  # (7,)

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
    args = parser.parse_args()

    result = predict_image(args.checkpoint, args.image, args.model)
    print(f"Predicted emotion: {result['label']} (label_id={result['label_id']})")
    print("Probabilities:")
    for emotion, prob in result["probabilities"].items():
        print(f"  {emotion}: {prob:.4f}")
