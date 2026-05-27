from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from src.predict import preprocess_image


def test_preprocess_image_returns_model_tensor(tmp_path: Path):
    image_path = tmp_path / "face.png"
    Image.fromarray(np.full((64, 64), 128, dtype=np.uint8)).save(image_path)

    tensor = preprocess_image(image_path)

    assert tensor.shape == (1, 1, 48, 48)


def test_preprocess_image_imagenet_norm_with_custom_size(tmp_path: Path):
    image_path = tmp_path / "face.png"
    Image.fromarray(np.full((64, 64), 128, dtype=np.uint8)).save(image_path)

    tensor = preprocess_image(image_path, image_size=224, imagenet_norm=True)

    assert tensor.shape == (1, 3, 224, 224)
