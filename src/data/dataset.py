import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import SPLIT_TO_USAGE
from src.data.prepare_data import _parse_label, _parse_pixels

_IMAGE_SIZE = 48


class FER2013Dataset(Dataset):
    """PyTorch Dataset for FER-2013 facial expression data."""

    def __init__(self, csv_path, split: str, transform=None):
        if split not in SPLIT_TO_USAGE:
            raise ValueError(f"Unknown split {split!r}. Valid split values are: {list(SPLIT_TO_USAGE)}")

        usage = SPLIT_TO_USAGE[split]
        df = pd.read_csv(csv_path)

        samples: list[tuple[np.ndarray, int]] = []
        for row in df.itertuples(index=False):
            row_data = row._asdict()
            if row_data.get("Usage") != usage:
                continue
            label = _parse_label(row_data.get("emotion"))
            pixels = _parse_pixels(row_data.get("pixels"))
            if label is None or pixels is None:
                continue
            # clip to [0, 255] and cast to uint8, then reshape to (48, 48)
            image = np.clip(pixels, 0, 255).astype(np.uint8).reshape(_IMAGE_SIZE, _IMAGE_SIZE)
            samples.append((image, label))

        self._samples = samples
        self._transform = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        image, label = self._samples[idx]
        if self._transform is not None:
            tensor = self._transform(image)
        else:
            tensor = torch.from_numpy(image)
        return tensor, label
