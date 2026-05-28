from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image

from src.config import EMOTION_LABELS
from src.data.transforms import get_eval_transforms
from src.models import create_model
from src.utils.checkpoints import load_checkpoint


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_emotion_model(
    checkpoint_path: str | Path,
    model_name: str = "efficientnet_b2",
    device: str = "auto",
) -> tuple[torch.nn.Module, torch.device]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    torch_device = get_device(device)

    model_kwargs = {}
    if model_name in {"resnet18", "efficientnet_b2"}:
        model_kwargs["weights"] = None
        model_kwargs["freeze_backbone"] = False

    model = create_model(model_name, **model_kwargs)
    load_checkpoint(checkpoint_path, model, map_location=torch_device)
    model.to(torch_device)
    model.eval()

    return model, torch_device


def preprocess_image_like(
    image,
    image_size: int = 224,
    imagenet_norm: bool = True,
) -> torch.Tensor:
    """Convert PIL/numpy image into a batch tensor for the model."""

    if isinstance(image, Image.Image):
        np_image = np.array(image.convert("L"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            np_image = image.astype(np.uint8)
        elif image.ndim == 3:
            np_image = np.array(Image.fromarray(image.astype(np.uint8)).convert("L"), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported numpy image shape: {image.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    transform = get_eval_transforms(image_size=image_size, imagenet_norm=imagenet_norm)
    return transform(np_image).unsqueeze(0)


@torch.no_grad()
def predict_emotion(
    model: torch.nn.Module,
    device: torch.device,
    image,
    image_size: int = 224,
    imagenet_norm: bool = True,
) -> dict:
    tensor = preprocess_image_like(
        image,
        image_size=image_size,
        imagenet_norm=imagenet_norm,
    ).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    label_id = int(probs.argmax())
    return {
        "label_id": label_id,
        "label": EMOTION_LABELS[label_id],
        "confidence": float(probs[label_id]),
        "probabilities": {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))},
        "probability_array": probs,
    }


def top_k(probabilities: Dict[str, float], k: int = 3) -> list[tuple[str, float]]:
    return sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:k]
